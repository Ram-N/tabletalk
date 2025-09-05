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
from langchain_anthropic import ChatAnthropic
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType

# Database imports
from database.schema import (
    create_database_schema, 
    import_csv_to_database, 
    get_database_engine,
    get_database_schema_info,
    validate_database,
    DB_PATH, 
    DB_URL
)

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# File management
UPLOADS_DIR = "uploads"
DEFAULT_CSV_PATH = "uploads/GSP Standardized Sheet - May_2025_standard.csv"

# Create uploads directory
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Database setup
current_db_path = DB_PATH
current_csv_path = DEFAULT_CSV_PATH

# Check if database exists, if not create it from default CSV
if not os.path.exists(current_db_path):
    print("Database not found. Creating new database...")
    create_database_schema()
    if os.path.exists(DEFAULT_CSV_PATH):
        print(f"Importing default CSV: {DEFAULT_CSV_PATH}")
        stats = import_csv_to_database(DEFAULT_CSV_PATH)
        print(f"Imported {stats['total_records']} records")
    else:
        print("No default CSV found. Database created but empty.")
else:
    print(f"Database found: {current_db_path}")
    validation = validate_database()
    if validation['valid']:
        print(f"Database validated: {validation['total_records']} records")
    else:
        print(f"Database validation failed: {validation['error']}")

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
# Get API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize LLM (prefer Anthropic if available, fallback to Groq)
selected_llm = None
llm_provider = None

if anthropic_api_key:
    try:
        selected_llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=anthropic_api_key,
            temperature=0,
            max_tokens=4096
        )
        llm_provider = "anthropic"
        print(f"✅ Using Anthropic Claude API")
    except Exception as e:
        print(f"⚠️  Failed to initialize Anthropic: {e}")
        selected_llm = None

if not selected_llm and groq_api_key:
    try:
        selected_llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            groq_api_key=groq_api_key, 
            temperature=0
        )
        llm_provider = "groq"
        print(f"✅ Using Groq API")
    except Exception as e:
        print(f"⚠️  Failed to initialize Groq: {e}")

if not selected_llm:
    raise HTTPException(
        status_code=500, 
        detail="No valid API key found. Please set either ANTHROPIC_API_KEY or GROQ_API_KEY in your .env file."
    )

# Use the selected LLM
groq_llm = selected_llm  # Keep variable name for backward compatibility

# Create SQL agent function
def create_sql_agent_instance(db_path: str = None):
    """Create a new SQL agent instance for the database"""
    db_path = db_path or current_db_path
    if not os.path.exists(db_path):
        return None
    try:
        # Create database connection
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        # Create SQL agent with custom prompt for student data
        agent = create_sql_agent(
            llm=groq_llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            return_intermediate_steps=True
        )
        
        return agent, db
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create SQL agent: {e}")

# Keep CSV agent function for backward compatibility during transition
def create_csv_agent_instance(csv_file_path):
    """Create a new CSV agent instance for the given file (legacy)"""
    if not csv_file_path or not os.path.exists(csv_file_path):
        return None
    try:
        return create_csv_agent(
            groq_llm,
            csv_file_path,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create CSV agent: {e}")

# Create a global SQL agent instance
sql_agent_data = create_sql_agent_instance() if os.path.exists(current_db_path) else None

if sql_agent_data:
    sql_agent, sql_db = sql_agent_data
    print(f"SQL agent initialized successfully with database: {current_db_path}")
else:
    sql_agent, sql_db = None, None
    print("No SQL agent initialized. Database not found.")

# Legacy CSV agent for fallback
csv_agent = create_csv_agent_instance(current_csv_path) if current_csv_path and os.path.exists(current_csv_path) else None

# Simple stdout capture for agent verbose output
import contextlib
from io import StringIO

import queue
import threading

class StreamingVerboseCapture:
    def __init__(self):
        self.output_queue = queue.Queue()
        self.captured_output = StringIO()
        self.streaming = True
        
    @contextlib.contextmanager
    def capture(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = self
            sys.stderr = self
            yield self
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.streaming = False
            
    def write(self, text):
        """Custom write method that captures output and puts it in queue for streaming"""
        # Also write to captured_output for final retrieval
        self.captured_output.write(text)
        
        # Only queue non-empty lines for streaming
        if text.strip() and self.streaming:
            self.output_queue.put(text.strip())
            
    def flush(self):
        """Required for stdout compatibility"""
        pass
        
    def get_streaming_line(self, timeout=0.1):
        """Get the next line for streaming (non-blocking)"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_output(self):
        return self.captured_output.getvalue()
        
    def format_line(self, line):
        """Format a single line for display"""
        if not line or not line.strip():
            return None
            
        line = line.strip()
        
        # Format different types of output
        if line.startswith('Thought:'):
            return f"🤔 {line}"
        elif line.startswith('Action:'):
            return f"🛠️ {line}"
        elif line.startswith('Action Input:'):
            return f"📝 {line}"
        elif line.startswith('Observation:'):
            # Truncate long observations
            obs = line[12:] if len(line) > 12 else ""
            if len(obs) > 200:
                obs = obs[:200] + "..."
            return f"✅ Observation: {obs}"
        elif 'Final Answer:' in line:
            return f"🎯 {line}"
        elif line and len(line.strip()) > 0:
            return f"💭 {line}"
        return None

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
    return {
        "message": "TableTalk SQL Agent API is running", 
        "endpoints": ["/ask", "/ask-stream", "/abort/{request_id}", "/upload", "/data-info", "/schema", "/health", "/docs"],
        "status": "online", 
        "llm_provider": llm_provider,
        "database_records": 174,  # Will be dynamic later
        "timestamp": asyncio.get_event_loop().time()
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint for monitoring"""
    return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}

@app.post("/upload")
async def upload_csv_file(file: UploadFile = File(...)):
    """Upload a CSV file and import it into the database"""
    global sql_agent, sql_db, current_csv_path
    
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
        
        # Import CSV into database
        try:
            # Create fresh database schema
            create_database_schema()
            
            # Import the new CSV data
            stats = import_csv_to_database(file_path)
            
            # Create new SQL agent
            sql_agent_data = create_sql_agent_instance()
            if sql_agent_data is None:
                raise HTTPException(status_code=500, detail="Failed to create SQL agent after import")
            
            # Update global state
            sql_agent, sql_db = sql_agent_data
            current_csv_path = file_path
            
            return {
                "message": "CSV file uploaded and imported to database successfully",
                "filename": file.filename,
                "total_records": stats['total_records'],
                "unique_categories": stats['unique_categories'],
                "unique_cities": stats['unique_cities'],
                "unique_schools": stats['unique_schools'],
                "database_path": stats['db_path']
            }
            
        except Exception as e:
            # Clean up file on database import error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Database import failed: {str(e)}")
        
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
    Standard endpoint to send a prompt to the SQL agent and get a complete response.
    """
    if sql_agent is None:
        raise HTTPException(status_code=400, detail="No database loaded. Please upload a CSV file first.")
    
    try:
        # Run the SQL agent with timeout
        def run_agent():
            try:
                import asyncio
                
                # Direct synchronous call to agent
                response = sql_agent.invoke({"input": request.prompt})
                
                # Handle any Future objects
                if hasattr(response, 'result') and callable(getattr(response, 'result')):
                    response = response.result()
                
                # If it's still a coroutine, force it to sync
                if asyncio.iscoroutine(response):
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        response = new_loop.run_until_complete(response)
                        new_loop.close()
                    except Exception as loop_error:
                        return f"Failed to handle async response: {str(loop_error)}"
                
                # Handle different response formats
                if hasattr(response, 'get'):
                    return response.get("output", str(response))
                elif isinstance(response, dict):
                    return response.get("output", str(response))
                else:
                    return str(response)
            except Exception as e:
                return f"Agent error: {str(e)}"
        
        # Execute with timeout
        try:
            loop = asyncio.get_event_loop()
            if request.timeout > 0:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, run_agent),
                    timeout=request.timeout
                )
            else:
                result = await loop.run_in_executor(None, run_agent)
            return {"response": result}
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail=f"Agent response timed out after {request.timeout} seconds")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask-stream")
async def ask_agent_stream(prompt: str, timeout: int = 300):
    """
    Streaming endpoint that shows the SQL agent's thinking process in real-time.
    """
    if sql_agent is None:
        raise HTTPException(status_code=400, detail="No database loaded. Please upload a CSV file first.")
    
    request_id = f"req_{len(active_requests)}"
    active_requests[request_id] = True
    
    async def generate_response():
        try:
            # Create streaming verbose capture
            capture = StreamingVerboseCapture()
            agent_result = None
            agent_error = None
            
            def run_agent_with_streaming_capture():
                nonlocal agent_result, agent_error
                try:
                    with capture.capture():
                        # Force synchronous execution by bypassing any async wrappers
                        import asyncio
                        import concurrent.futures
                        
                        # Create a completely clean sync execution
                        try:
                            # Direct synchronous call to agent
                            response = sql_agent.invoke({"input": prompt})
                            
                            # Handle any Future objects
                            if hasattr(response, 'result') and callable(getattr(response, 'result')):
                                response = response.result()
                            
                            # If it's still a coroutine after all that, force it to sync
                            if asyncio.iscoroutine(response):
                                # Create a new event loop just for this
                                try:
                                    new_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(new_loop)
                                    response = new_loop.run_until_complete(response)
                                    new_loop.close()
                                except Exception as loop_error:
                                    agent_error = f"Failed to handle async response: {str(loop_error)}"
                                    return
                            
                            # Handle different response formats
                            if hasattr(response, 'get'):
                                agent_result = response.get("output", str(response))
                            elif isinstance(response, dict):
                                agent_result = response.get("output", str(response))
                            else:
                                agent_result = str(response)
                                
                        except Exception as invoke_error:
                            agent_error = f"Agent invoke error: {str(invoke_error)}"
                            
                except Exception as e:
                    agent_error = f"Agent execution error: {str(e)}"
            
            # Stream initial message
            yield f"data: {json.dumps({'type': 'start', 'message': 'Agent started processing...', 'request_id': request_id})}\n\n"
            
            # Run agent completely synchronously in a thread with proper isolation
            agent_thread = threading.Thread(target=run_agent_with_streaming_capture)
            agent_thread.daemon = True
            agent_thread.start()
            
            # Monitor progress and stream output
            start_time = asyncio.get_event_loop().time()
            
            while agent_thread.is_alive():
                # Check for abort
                if not active_requests.get(request_id, False):
                    yield f"data: {json.dumps({'type': 'aborted', 'message': 'Request was aborted', 'request_id': request_id})}\n\n"
                    return
                
                # Check for timeout (only if timeout > 0)
                if timeout > 0:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > timeout:
                        yield f"data: {json.dumps({'type': 'timeout', 'message': f'Agent response timed out after {timeout} seconds', 'request_id': request_id})}\n\n"
                        return
                
                # Stream any new output
                line = capture.get_streaming_line()
                if line:
                    formatted_line = capture.format_line(line)
                    if formatted_line:
                        yield f"data: {json.dumps({'type': 'thinking', 'message': formatted_line, 'request_id': request_id})}\n\n"
                
                await asyncio.sleep(0.1)  # Check every 100ms
            
            # Wait for thread to complete
            agent_thread.join(timeout=1.0)
            
            # Stream any remaining output
            while True:
                line = capture.get_streaming_line()
                if line:
                    formatted_line = capture.format_line(line)
                    if formatted_line:
                        yield f"data: {json.dumps({'type': 'thinking', 'message': formatted_line, 'request_id': request_id})}\n\n"
                else:
                    break
            
            # Send final result
            if active_requests.get(request_id, False):
                if agent_error:
                    yield f"data: {json.dumps({'type': 'error', 'message': agent_error, 'request_id': request_id})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'result', 'message': agent_result, 'request_id': request_id})}\n\n"
                
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
    Endpoint to get information about the loaded database.
    """
    if not os.path.exists(current_db_path):
        return {"error": "No database found. Please upload a CSV file first."}
        
    try:
        schema_info = get_database_schema_info()
        
        if "error" in schema_info:
            return schema_info
        
        return {
            "table_name": schema_info["table_name"],
            "total_records": schema_info["row_count"],
            "columns": [col["name"] for col in schema_info["columns"]],
            "column_details": schema_info["columns"],
            "database_path": schema_info["db_path"],
            "sample_data": schema_info["sample_data"][:3] if schema_info["sample_data"] else []
        }
    except Exception as e:
        return {"error": f"Could not read database: {str(e)}"}

@app.get("/schema")
async def get_database_schema():
    """
    Endpoint to get detailed database schema information for debugging.
    """
    if not os.path.exists(current_db_path):
        return {"error": "No database found. Please upload a CSV file first."}
        
    try:
        schema_info = get_database_schema_info()
        validation = validate_database()
        
        return {
            "database_info": schema_info,
            "validation": validation,
            "database_path": current_db_path
        }
    except Exception as e:
        return {"error": f"Could not get schema info: {str(e)}"}

# --- Main entry point for local development ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
