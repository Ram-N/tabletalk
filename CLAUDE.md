# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management
- Always use `uv` for Python package management
- Commands:
  - `uv pip install -r requirements.txt` - Install dependencies  
  - `uv run python main.py` - Run the FastAPI backend server
  - `uv run uvicorn main:app --reload` - Run with hot reload for development

## Project Overview

TableTalk is a LangChain CSV agent powered conversational AI for analyzing CSV data. It provides a FastAPI backend with streaming capabilities connected to an interactive HTML frontend. The application supports file upload, real-time streaming of AI reasoning, and dynamic CSV processing.

## Tech Stack
- **Backend**: FastAPI with Python 3.12+ and Server-Sent Events for streaming
- **AI Framework**: LangChain experimental CSV agent with Groq LLM (llama-3.1-8b-instant)
- **Data Processing**: CSV analysis using pandas integration with file upload support
- **Frontend**: Vanilla HTML/JavaScript with Tailwind CSS and real-time streaming
- **Features**: File upload, timeout control, abort functionality, health monitoring, and thinking process visualization

## Architecture

The application follows a client-server architecture with streaming capabilities:

1. **Frontend** (`index.html`): Interactive web interface with real-time agent thinking display
2. **Backend** (`main.py`): FastAPI server with `/ask` and `/ask-stream` endpoints
3. **Agent System**: LangChain CSV agent with ChatGroq LLM for data analysis
4. **Streaming**: Server-Sent Events for real-time thinking process visualization

## Key Components

- **CSV Agent**: Uses `create_csv_agent` from LangChain experimental with ChatGroq LLM (main.py:83-96)
- **File Upload**: Dynamic CSV file processing with validation (main.py:183-248)  
- **Streaming Support**: Real-time display of agent's thought process via Server-Sent Events (main.py:279-346)
- **Health Monitoring**: Automatic server status checks and crash detection (index.html:439-497)
- **Timeout Control**: Configurable timeout with countdown and abort functionality
- **Error Handling**: Comprehensive error handling for timeouts, agent failures, and file upload issues
- **CORS**: Configured for frontend-backend communication

## API Endpoints

- `GET /` - Serve the HTML frontend
- `POST /upload` - Upload CSV file and create new agent instance
- `POST /ask` - Standard synchronous agent query
- `GET /ask-stream` - Streaming agent query with thinking process
- `POST /abort/{request_id}` - Cancel running streaming request
- `GET /data-info` - Get current CSV file metadata
- `GET /health` - Server health check
- `GET /api` - API information

## File Structure

- `main.py` - FastAPI backend with all API endpoints and CSV agent logic
- `index.html` - Complete frontend with file upload, streaming, and health monitoring
- `uploads/` - Directory for user uploaded CSV files (auto-created)
- `docs/` - Comprehensive documentation including developer guide

## Environment Variables

Required API keys in `.env` file:
- `GROQ_API_KEY` - For the Groq LLM service used by the CSV agent

## Development Commands

- `uv run python main.py` - Start server (basic mode)
- `uv run uvicorn main:app --reload` - Start with hot reload (recommended for development)
- Server runs on `http://localhost:8000` by default
- Frontend served at root `/`, API docs at `/docs`

## Important Implementation Details

- **Agent Creation**: New agent instance created per uploaded file (main.py:83-96)
- **Streaming Capture**: Uses VerboseCapture class to capture agent's verbose output (main.py:110-161)
- **File Upload Validation**: Size limit (10MB), CSV format validation, pandas parsing check
- **Server Health**: Frontend monitors server every 10 seconds, handles crashes gracefully
- **Request Management**: Active request tracking with abort functionality via request IDs

## Security Note

The `.env` file contains API keys and should never be committed. It's properly gitignored.