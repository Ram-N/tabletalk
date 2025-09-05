# Getting Started with TableTalk

TableTalk is an AI-powered conversational interface that combines the power of Groq's LLM with your Excel data and web search capabilities. Ask questions in natural language about your data or any topic, and get intelligent responses.

## Prerequisites

- Python 3.12+
- `uv` package manager
- API keys for Groq and Tavily (see setup below)

## Setup

### 1. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with your API keys:

```env
GROQ_API_KEY="your_groq_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
```

### 3. Start the Application

```bash
uv run python main.py
```

The server will start on `http://localhost:8000`

### 4. Access the Web Interface

Open `index.html` in your browser or serve it from a local web server to interact with the agent.

## Querying Your Excel Data

TableTalk automatically loads the Excel file from `data/GSP Standardized Sheet.xlsx` at startup. You can query this data using natural language.

### Example Data Queries

**Basic Data Information:**
- "What data do you have?"
- "How many rows are in the dataset?"
- "What columns are available?"

**Data Analysis:**
- "Show me the first few rows of data"
- "What are the unique values in the [column_name] column?"
- "Calculate the average of [column_name]"
- "Find records where [column_name] equals [value]"

**Data Insights:**
- "Summarize the key findings from this dataset"
- "What trends do you see in the data?"
- "Are there any outliers or anomalies?"

### How It Works

1. **Smart Context**: When your question contains data-related keywords (data, excel, sheet, table, rows, columns), the agent automatically includes relevant context about your Excel file.

2. **Column Awareness**: The agent knows your column names and can reference them in responses.

3. **Sample Data**: For data-related queries, the agent receives the first few rows as context to provide accurate answers.

4. **Fallback to Search**: For questions unrelated to your data, the agent uses Tavily web search to find current information.

## API Endpoints

### POST `/ask`
Send queries to the agent:
```json
{
  "prompt": "What data do you have available?"
}
```

### GET `/data-info`
Get metadata about the loaded Excel file:
```json
{
  "rows": 100,
  "columns": ["Column1", "Column2", "Column3"],
  "sample_data": {...}
}
```

## Tips for Better Results

- **Be specific**: Instead of "analyze data," try "show me the distribution of values in the status column"
- **Ask follow-ups**: The agent maintains conversation history, so you can ask follow-up questions
- **Combine queries**: You can ask about both your data and external information in the same conversation
- **Use natural language**: No need for SQL or technical syntax - just ask naturally

## Troubleshooting

- **Excel not loading**: Check that `data/GSP Standardized Sheet.xlsx` exists
- **API errors**: Verify your `.env` file contains valid API keys
- **Connection issues**: Ensure the backend is running on port 8000 before accessing the frontend