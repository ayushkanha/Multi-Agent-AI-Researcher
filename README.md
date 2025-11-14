# Multi-Agent AI Researcher

A sophisticated Streamlit application that leverages a collaborative multi-agent system to conduct in-depth research on any given topic and generate comprehensive, publication-ready reports.

---

<!-- Add a screenshot or GIF of the application in action -->
*<p align="center">![alt text](image-1.png)</p>*

## 🚀 Features

- **Multi-Agent Collaboration**: Utilizes two parallel research agents to gather diverse information and avoid overlapping search results.
- **Deep Research Capability**: Agents are prompted to perform methodical, in-depth research, not just superficial summaries.
- **Automated Report Generation**: A dedicated report-writing agent synthesizes the findings into a structured, professional markdown document.
- **Parallel Processing**: Employs a `ThreadPoolExecutor` to run research agents concurrently for faster results.
- **Multiple LLM Integration**: Uses Google's Gemini models for research and Groq's Llama model for fast report generation.
- **PDF Export**: Instantly download the final, formatted research report as a PDF.
- **Customizable UI**: A clean and intuitive interface built with Streamlit, with custom CSS for styling.

## 🏛️ Project Architecture
agent.png
The application follows a three-stage pipeline:

1.  **Input**: The user provides a research query through the Streamlit interface.
2.  **Research (Multi-Agent)**:
    - Two independent "Research Agents" (powered by Gemini 2.0 Flash) are spawned.
    - Each agent is assigned a unique ID and uses the DuckDuckGo search tool (`internet_search`) to find information.
    - The search results are partitioned between the agents to ensure they gather unique information, promoting wider coverage.
    - The agents run in parallel to speed up the information-gathering process.
3.  **Synthesis (Report Agent)**:
    - The findings from both research agents are collected and combined.
    - A specialized "Report Writer Agent" (powered by Llama on Groq) receives the combined research.
    - This agent follows a detailed system prompt to structure, format, and write a comprehensive report in Markdown, complete with an executive summary, detailed sections, analysis, and references.
4.  **Output**:
    - The final report is displayed in the Streamlit app.
    - A PDF version of the report is generated and made available for download.



## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI & LLMs**:
  - LangChain for agent creation and orchestration.
  - Google Gemini for research agents.
  - Groq with Meta Llama for the report-writing agent.
- **Tools**:
  - DuckDuckGo Search for the internet search tool.
  - Markdown-PDF for PDF generation.
- **Language**: Python

## ⚙️ Setup and Installation

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
streamlit
langchain-core
duckduckgo-search
deepagents
langchain-google-genai
langchain-groq
langsmith
python-dotenv
markdown-pdf
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory of the project and add your API keys. You will need two separate Gemini API keys to run the agents in parallel without rate-limiting issues.

```env
Groq_api_key="YOUR_GROQ_API_KEY"
Gemini_api_key1="YOUR_FIRST_GEMINI_API_KEY"
Gemini_api_key2="YOUR_SECOND_GEMINI_API_KEY"
```

### 5. Create the CSS File

Create a `style.css` file in the root directory to add custom styles for the Streamlit app. You can start with the example in the `app.py` or create your own.

### 6. Run the Application

Execute the following command in your terminal:

```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501`.

## 📖 How to Use

1.  Enter your research topic or question in the text input field (e.g., "What is LangGraph?").
2.  Click the "Run Research" button.
3.  Wait for the agents to complete their research and for the final report to be generated.
4.  View the report directly in the app or click the "Download PDF Report" button to save it locally.
