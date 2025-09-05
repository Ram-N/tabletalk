import os
import uvicorn
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
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

# CSV file path
CSV_FILE_PATH = "data/dataset.csv"

# Check if CSV file exists
if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE_PATH}")

print(f"CSV file found: {CSV_FILE_PATH}")

# Set up Pydantic models for request bodies
class PromptRequest(BaseModel):
    prompt: str = Field(..., description="The user's query or instruction.")
    timeout: int = Field(60, description="Timeout in seconds for the agent response.")

class StreamPromptRequest(BaseModel):
    prompt: str = Field(..., description="The user's query or instruction.")
    timeout: int = Field(60, description="Timeout in seconds for the agent response.")

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
def create_csv_agent_instance():
    """Create a new CSV agent instance"""
    try:
        return create_csv_agent(
            groq_llm,
            CSV_FILE_PATH,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True  # Required for CSV agent to execute pandas code
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create CSV agent: {e}")

# Create a global agent instance
csv_agent = create_csv_agent_instance()

print(f"CSV agent initialized successfully with file: {CSV_FILE_PATH}")

# Helper function to capture agent output
class AgentOutputCapture:
    def __init__(self):
        self.outputs = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        self.string_io = io.StringIO()
        sys.stdout = self.string_io
        sys.stderr = self.string_io
        return self
        
    def __exit__(self, type, value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
    def get_output(self):
        return self.string_io.getvalue()

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
    return {"message": "TableTalk CSV Agent API is running", "endpoints": ["/ask", "/ask-stream", "/abort/{request_id}", "/data-info", "/docs"]}

@app.post("/ask")
async def ask_agent(request: PromptRequest):
    """
    Standard endpoint to send a prompt to the CSV agent and get a complete response.
    """
    try:
        # Run the CSV agent with timeout
        def run_agent():
            return csv_agent.run(request.prompt)
        
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
async def ask_agent_stream(prompt: str, timeout: int = 60):
    """
    Streaming endpoint that shows the agent's thinking process in real-time.
    """
    request_id = f"req_{len(active_requests)}"
    active_requests[request_id] = True
    
    async def generate_response():
        try:
            # Capture the agent's verbose output
            def run_agent_with_capture():
                with AgentOutputCapture() as capture:
                    try:
                        result = csv_agent.run(prompt)
                        return result, capture.get_output()
                    except Exception as e:
                        return str(e), capture.get_output()
            
            # Run agent in thread with timeout
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, run_agent_with_capture)
            
            # Stream initial message
            yield f"data: {json.dumps({'type': 'start', 'message': 'Agent started processing...', 'request_id': request_id})}\n\n"
            
            try:
                result, captured_output = await asyncio.wait_for(future, timeout=timeout)
                
                # Parse and stream the captured output
                if captured_output:
                    lines = captured_output.split('\n')
                    for line in lines:
                        if line.strip() and not active_requests.get(request_id, False):
                            break
                        if line.strip():
                            yield f"data: {json.dumps({'type': 'thinking', 'message': line.strip(), 'request_id': request_id})}\n\n"
                            await asyncio.sleep(0.1)  # Small delay for better UX
                
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
    try:
        import pandas as pd
        df = pd.read_csv(CSV_FILE_PATH)
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "file_path": CSV_FILE_PATH,
            "sample_data": df.head().to_dict()
        }
    except Exception as e:
        return {"error": f"Could not read CSV file: {str(e)}"}

# --- Main entry point for local development ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
