# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management
- Always use `uv` for Python package management
- Commands:
  - `uv pip install -r requirements.txt` - Install dependencies  
  - `uv run python main.py` - Run the FastAPI backend server
  - `uv run uvicorn main:app --reload` - Run with hot reload for development

## Project Overview

TableTalk is a LangChain CSV agent powered conversational AI for analyzing CSV data. It provides a FastAPI backend with streaming capabilities connected to an interactive HTML frontend.

## Tech Stack
- **Backend**: FastAPI with Python 3.12+ and Server-Sent Events for streaming
- **AI Framework**: LangChain with Groq LLM (llama-3.1-8b-instant) and CSV agent
- **Data Processing**: CSV analysis using pandas integration
- **Frontend**: Vanilla HTML/JavaScript with Tailwind CSS and real-time streaming
- **Features**: Timeout control, abort functionality, and thinking process visualization

## Architecture

The application follows a client-server architecture with streaming capabilities:

1. **Frontend** (`index.html`): Interactive web interface with real-time agent thinking display
2. **Backend** (`main.py`): FastAPI server with `/ask` and `/ask-stream` endpoints
3. **Agent System**: LangChain CSV agent with ChatGroq LLM for data analysis
4. **Streaming**: Server-Sent Events for real-time thinking process visualization

## Key Components

- **CSV Agent**: Uses `create_csv_agent` with ChatGroq LLM for data analysis
- **Streaming Support**: Real-time display of agent's thought process via SSE
- **Timeout Control**: Configurable timeout with countdown and abort functionality
- **Error Handling**: Comprehensive error handling for timeouts and agent failures
- **CORS**: Configured for frontend-backend communication

## Environment Variables

Required API keys in `.env` file:
- `GROQ_API_KEY` - For the Groq LLM service used by the CSV agent

## Development Server

The FastAPI server runs on `localhost:8000` by default. Frontend connects to this endpoint for API communication.

## Security Note

The `.env` file contains API keys and should never be committed. It's properly gitignored.