import os
import uvicorn
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import io
import sys

# LangChain imports
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents import AgentType

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# CSV file management
UPLOADS_DIR = "uploads"
DEFAULT_CSV_PATH = "data/dataset.csv"
current_csv_path = DEFAULT_CSV_PATH

# Create uploads directory
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Check if default CSV file exists
if os.path.exists(DEFAULT_CSV_PATH):
    print(f"Default CSV file found: {DEFAULT_CSV_PATH}")
else:
    current_csv_path = None
    print("No default CSV file found. Please upload a CSV file to get started.")

# Set up Pydantic models for request bodies
class PromptRequest(BaseModel):
    prompt: str = Field(..., description="The user's query or instruction.")
    timeout: int = Field(300, description="Timeout in seconds for the agent response.")

class StreamPromptRequest(BaseModel):
    prompt: str = Field(..., description="The user's query or instruction.")
    timeout: int = Field(300, description="Timeout in seconds for the agent response.")

# Global variables for request cancellation
active_requests = {}

# Initialize FastAPI app
app = FastAPI(
    title="TableTalk CSV Agent API",
    description="A CSV analysis agent powered by LangChain and Groq API with streaming capabilities."
)

# Configure CORS to allow the frontend to access the API
# NOTE: In a production environment, you should replace "*" with your specific frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Agent and Tool Setup ---
# Get API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set. Please add it to your .env file or environment variables.")

# Initialize the Groq LLM
try:
    groq_llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to initialize ChatGroq: {e}.")

# Create CSV agent function
def create_csv_agent_instance(csv_file_path):
    """Create a new CSV agent instance for the given file"""
    if not csv_file_path or not os.path.exists(csv_file_path):
        return None
    try:
        return create_csv_agent(
            groq_llm,
            csv_file_path,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True  # Required for CSV agent to execute pandas code
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create CSV agent: {e}")

# Create a global agent instance (if default file exists)
csv_agent = create_csv_agent_instance(current_csv_path) if current_csv_path else None

if csv_agent:
    print(f"CSV agent initialized successfully with file: {current_csv_path}")
else:
    print("No CSV agent initialized. Upload a file to create one.")

# Simple stdout capture for agent verbose output
import contextlib
from io import StringIO

class VerboseCapture:
    def __init__(self):
        self.captured_output = StringIO()
        
    @contextlib.contextmanager
    def capture(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = self.captured_output
            sys.stderr = self.captured_output
            yield self
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
    def get_output(self):
        return self.captured_output.getvalue()
        
    def get_formatted_steps(self):
        """Parse the captured output and format it nicely"""
        output = self.get_output()
        if not output:
            return []
            
        lines = output.split('\n')
        formatted_steps = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Format different types of output
            if line.startswith('Thought:'):
                formatted_steps.append(f"🤔 {line}")
            elif line.startswith('Action:'):
                formatted_steps.append(f"🛠️ {line}")
            elif line.startswith('Action Input:'):
                formatted_steps.append(f"📝 {line}")
            elif line.startswith('Observation:'):
                # Truncate long observations
                obs = line[12:] if len(line) > 12 else ""
                if len(obs) > 200:
                    obs = obs[:200] + "..."
                formatted_steps.append(f"✅ Observation: {obs}")
            elif 'Final Answer:' in line:
                formatted_steps.append(f"🎯 {line}")
            elif line and len(line.strip()) > 0:
                formatted_steps.append(f"💭 {line}")
                
        return formatted_steps

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML frontend"""
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return {"message": "TableTalk CSV Agent API is running", "endpoints": ["/ask", "/ask-stream", "/abort/{request_id}", "/data-info", "/docs"]}

@app.get("/api")
async def api_info():
    return {"message": "TableTalk CSV Agent API is running", "endpoints": ["/ask", "/ask-stream", "/abort/{request_id}", "/upload", "/data-info", "/health", "/docs"], "status": "online", "timestamp": asyncio.get_event_loop().time()}

@app.get("/health")
async def health_check():
    """Simple health check endpoint for monitoring"""
    return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}

@app.post("/upload")
async def upload_csv_file(file: UploadFile = File(...)):
    """Upload a CSV file and create a new agent for it"""
    global csv_agent, current_csv_path
    
    # Validate file type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Validate file size (max 10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 10MB")
    
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Validate CSV format
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
            if len(df) == 0:
                raise ValueError("CSV file is empty")
            if len(df.columns) == 0:
                raise ValueError("CSV file has no columns")
        except Exception as e:
            os.remove(file_path)  # Clean up invalid file
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        # Create new CSV agent
        new_agent = create_csv_agent_instance(file_path)
        if new_agent is None:
            os.remove(file_path)
            raise HTTPException(status_code=500, detail="Failed to create CSV agent")
        
        # Update global state
        old_csv_path = current_csv_path
        csv_agent = new_agent
        current_csv_path = file_path
        
        # Clean up old uploaded file (but not the default file)
        if old_csv_path and old_csv_path != DEFAULT_CSV_PATH and os.path.exists(old_csv_path):
            os.remove(old_csv_path)
        
        # Convert sample data to JSON-safe format (handle NaN values)
        sample_data = df.head().fillna("").to_dict()
        
        return {
            "message": "CSV file uploaded successfully",
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "sample_data": sample_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file on error
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/ask")
async def ask_agent(request: PromptRequest):
    """
    Standard endpoint to send a prompt to the CSV agent and get a complete response.
    """
    if csv_agent is None:
        raise HTTPException(status_code=400, detail="No CSV file loaded. Please upload a CSV file first.")
    
    try:
        # Run the CSV agent with timeout
        def run_agent():
            response = csv_agent.invoke({"input": request.prompt})
            return response.get("output", "No response generated")
        
        # Execute with timeout
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, run_agent),
                timeout=request.timeout
            )
            return {"response": result}
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail=f"Agent response timed out after {request.timeout} seconds")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask-stream")
async def ask_agent_stream(prompt: str, timeout: int = 300):
    """
    Streaming endpoint that shows the agent's thinking process in real-time.
    """
    if csv_agent is None:
        raise HTTPException(status_code=400, detail="No CSV file loaded. Please upload a CSV file first.")
    
    request_id = f"req_{len(active_requests)}"
    active_requests[request_id] = True
    
    async def generate_response():
        try:
            # Create verbose capture to get agent output
            def run_agent_with_capture():
                capture = VerboseCapture()
                try:
                    with capture.capture():
                        response = csv_agent.invoke({"input": prompt})
                        result = response.get("output", "No response generated")
                    return result, capture.get_formatted_steps()
                except Exception as e:
                    return str(e), capture.get_formatted_steps()
            
            # Run agent in thread with timeout
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, run_agent_with_capture)
            
            # Stream initial message
            yield f"data: {json.dumps({'type': 'start', 'message': 'Agent started processing...', 'request_id': request_id})}\n\n"
            
            try:
                result, captured_steps = await asyncio.wait_for(future, timeout=timeout)
                
                # Stream the captured steps
                if captured_steps:
                    for step in captured_steps:
                        if not active_requests.get(request_id, False):
                            break
                        if step.strip():
                            yield f"data: {json.dumps({'type': 'thinking', 'message': step.strip(), 'request_id': request_id})}\n\n"
                            await asyncio.sleep(0.3)  # Small delay for better readability
                
                # Send final result
                if active_requests.get(request_id, False):
                    yield f"data: {json.dumps({'type': 'result', 'message': result, 'request_id': request_id})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'aborted', 'message': 'Request was aborted', 'request_id': request_id})}\n\n"
                    
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'timeout', 'message': f'Agent response timed out after {timeout} seconds', 'request_id': request_id})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'request_id': request_id})}\n\n"
        finally:
            # Cleanup
            active_requests.pop(request_id, None)
            yield f"data: {json.dumps({'type': 'end', 'message': 'Stream ended', 'request_id': request_id})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.post("/abort/{request_id}")
async def abort_request(request_id: str):
    """
    Abort a running streaming request.
    """
    if request_id in active_requests:
        active_requests[request_id] = False
        return {"message": f"Request {request_id} aborted"}
    return {"message": f"Request {request_id} not found or already completed"}

@app.get("/data-info")
async def get_data_info():
    """
    Endpoint to get information about the loaded CSV data.
    """
    if current_csv_path is None or not os.path.exists(current_csv_path):
        return {"error": "No CSV file loaded. Please upload a CSV file first."}
        
    try:
        import pandas as pd
        df = pd.read_csv(current_csv_path)
        # Convert sample data to JSON-safe format (handle NaN values)
        sample_data = df.head().fillna("").to_dict()
        
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "file_path": current_csv_path,
            "filename": os.path.basename(current_csv_path),
            "sample_data": sample_data
        }
    except Exception as e:
        return {"error": f"Could not read CSV file: {str(e)}"}

# --- Main entry point for local development ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
