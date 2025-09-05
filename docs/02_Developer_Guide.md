# Developer Guide - TableTalk CSV Agent

## 📋 Overview

TableTalk is a CSV data analysis application powered by LangChain and Groq AI. It features file upload, real-time streaming of AI reasoning, and interactive data exploration through natural language queries.

## 🏗️ Architecture

### Tech Stack
- **Backend**: FastAPI with Python 3.12+
- **AI Framework**: LangChain with Groq LLM (llama-3.1-8b-instant)
- **Frontend**: Vanilla HTML/JavaScript with TailwindCSS
- **Package Manager**: `uv` (modern Python package management)
- **Data Processing**: pandas for CSV handling

### Key Components
- **CSV Agent**: LangChain experimental CSV agent with pandas integration
- **Streaming API**: Server-Sent Events for real-time AI thinking process
- **File Upload**: Dynamic CSV file processing with validation
- **Health Monitoring**: Automatic server status monitoring

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/))
- Git
- A code editor (VS Code recommended)

### 1. Clone Repository
```bash
git clone <repository-url>
cd tabletalk
```

### 2. Environment Setup
```bash
# Install dependencies using uv
uv pip install -r requirements.txt

# Create .env file with required API keys
cp .env.example .env
```

### 3. Configure Environment Variables
Edit `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

**Get Groq API Key:**
1. Visit [console.groq.com](https://console.groq.com)
2. Create account and generate API key
3. Add to `.env` file

### 4. Run the Application
```bash
# Start the FastAPI server
uv run python main.py

# OR with hot reload for development
uv run uvicorn main:app --reload
```

### 5. Access the Application
- **Frontend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
tabletalk/
├── main.py                    # FastAPI backend server
├── index.html                 # Frontend interface
├── requirements.txt           # Python dependencies
├── CLAUDE.md                 # AI assistant instructions
├── .env                      # Environment variables (create this)
├── .gitignore               # Git ignore rules
├── data/                    # CSV data files (gitignored)
│   ├── dataset.csv         # Default dataset
│   └── *.csv              # Generated CSV files from Excel tabs
├── uploads/                 # User uploaded files (auto-created)
└── docs/                   # Documentation
    ├── 01_Getting_started.md
    └── 02_Developer_Guide.md
```

## 🔧 Development Workflow

### Running in Development Mode
```bash
# With auto-reload (recommended for development)
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# OR simple mode
uv run python main.py
```

### Installing New Dependencies
```bash
# Add new package
uv pip install package-name

# Update requirements.txt
uv pip freeze > requirements.txt
```

### Code Style and Standards
- **Python**: Follow PEP 8 conventions
- **JavaScript**: Use modern ES6+ features
- **HTML**: Semantic HTML5 with TailwindCSS classes
- **API**: RESTful design with clear endpoint naming

### Testing File Upload
```bash
# Test upload endpoint directly
curl -X POST http://localhost:8000/upload -F "file=@data/dataset.csv"

# Test health check
curl http://localhost:8000/health
```

## 🔌 API Endpoints

### Core Endpoints
- `GET /` - Serve frontend HTML
- `POST /upload` - Upload CSV file
- `POST /ask` - Standard AI query
- `GET /ask-stream` - Streaming AI query with thinking process
- `POST /abort/{request_id}` - Cancel running request
- `GET /data-info` - Get current CSV file information
- `GET /health` - Server health check
- `GET /api` - API information

### Request/Response Examples

**Upload CSV File:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@path/to/your/data.csv"
```

**Ask Question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How many rows are in the dataset?", "timeout": 300}'
```

**Stream Thinking Process:**
```bash
curl "http://localhost:8000/ask-stream?prompt=Analyze%20the%20data&timeout=300"
```

## 🐛 Common Issues and Solutions

### Server Keeps Crashing
**Symptoms**: Red status indicator, "Server Offline" message
**Causes**: 
- Large CSV files consuming too much memory
- Complex AI queries timing out
- Groq API rate limiting

**Solutions**:
```bash
# Restart server
uv run python main.py

# Check server logs for specific errors
# Reduce CSV file size or simplify queries
# Verify Groq API key is valid
```

### File Upload Fails
**Check**: 
- File size < 10MB
- File extension is `.csv`
- CSV format is valid
- `python-multipart` is installed

### Dependencies Issues
```bash
# Reinstall all dependencies
uv pip install -r requirements.txt

# Check for missing packages
uv pip list
```

## 💡 Key Features Implementation

### 1. CSV Agent Integration
Location: `main.py:82-96`
```python
def create_csv_agent_instance(csv_file_path):
    return create_csv_agent(
        groq_llm,
        csv_file_path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True  # Required for pandas operations
    )
```

### 2. Real-time Streaming
Location: `main.py:274-347`
- Uses Server-Sent Events (SSE)
- Captures agent's verbose thinking process
- Streams Thought → Action → Observation cycle

### 3. File Upload & Validation
Location: `main.py:178-240`
- Validates file type, size, and CSV format
- Creates new agent instance per file
- Handles NaN values for JSON serialization

### 4. Health Monitoring
Location: `index.html:433-492`
- Checks server every 10 seconds
- Automatic UI updates on server status changes
- Graceful error handling

## 🔄 Extending the Application

### Adding New Features

1. **New API Endpoint**:
```python
@app.post("/new-feature")
async def new_feature(request: NewRequest):
    # Implementation here
    return {"result": "success"}
```

2. **Frontend Integration**:
```javascript
async function callNewFeature() {
    const response = await fetch('http://localhost:8000/new-feature', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({data: 'value'})
    });
    return await response.json();
}
```

3. **Update Requirements**:
```bash
uv pip install new-package
uv pip freeze > requirements.txt
```

### Performance Optimization Tips

1. **Reduce CSV file size** - Large files slow down the agent
2. **Optimize queries** - Simple questions process faster
3. **Implement caching** - Cache frequent query results
4. **Use async operations** - For better concurrent handling
5. **Monitor memory usage** - Agent can be memory-intensive

## 🏭 Production Deployment

### Environment Variables
```bash
GROQ_API_KEY=prod_api_key
UPLOAD_MAX_SIZE=10485760  # 10MB
CORS_ORIGINS=https://yourdomain.com
```

### Security Considerations
- Validate all file uploads
- Implement rate limiting
- Use HTTPS in production
- Sanitize user inputs
- Monitor server resources

### Docker Deployment (Optional)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📝 Development Notes

### Important Files to Understand
1. **`main.py`** - Core backend logic, API endpoints, CSV agent setup
2. **`index.html`** - Frontend UI, streaming implementation, file upload
3. **`CLAUDE.md`** - AI assistant instructions and project context
4. **`requirements.txt`** - Python dependencies

### Code Patterns
- **Error Handling**: All API calls have try/catch with user-friendly messages
- **Async Operations**: Use async/await for I/O operations
- **Streaming**: Server-Sent Events for real-time updates
- **Validation**: Multiple layers of validation for uploads and queries

### Testing Strategy
1. **Manual Testing**: Upload various CSV files, test different query types
2. **API Testing**: Use curl or Postman for endpoint testing
3. **Error Testing**: Test server crashes, invalid files, timeouts
4. **Browser Testing**: Test UI across different browsers

## 🤝 Contributing Guidelines

1. **Fork repository** and create feature branch
2. **Follow code style** - consistent with existing code
3. **Test thoroughly** - manual testing of new features
4. **Update documentation** - update this guide if needed
5. **Create pull request** with clear description

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "feat: add new feature description"

# Push and create PR
git push origin feature/new-feature
```

## 📞 Support

### Getting Help
- **Documentation**: Check `docs/` folder
- **Issues**: Look at common issues section above
- **API Docs**: Visit http://localhost:8000/docs when server is running
- **Logs**: Check terminal output for detailed error messages

### Useful Commands Reference
```bash
# Development
uv run python main.py              # Start server
uv run uvicorn main:app --reload   # Start with auto-reload
uv pip install package-name        # Install new package
uv pip freeze > requirements.txt   # Update requirements

# Testing  
curl http://localhost:8000/health  # Health check
curl http://localhost:8000/api     # API info

# Debugging
lsof -ti:8000 | xargs kill -9      # Kill server on port 8000
ps aux | grep python               # Check running Python processes
```

---

**Happy Coding! 🚀**

*Last updated: [Current Date]*
*For questions or issues, refer to the project repository or contact the development team.*