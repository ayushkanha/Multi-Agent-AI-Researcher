import streamlit as st
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from langchain_core.messages import AIMessage
from ddgs import DDGS
from deepagents import create_deep_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langsmith.run_helpers import traceable
from dotenv import load_dotenv
from markdown_pdf import MarkdownPdf, Section

load_dotenv()

Groq_api_key = os.getenv("Groq_api_key")
Gemini_api_key1 = os.getenv("Gemini_api_key1")
Gemini_api_key2 = os.getenv("Gemini_api_key2")

# ----------------------------- CUSTOM CSS -----------------------------
def load_custom_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------------ TOOL FUNCTION ------------------------------
@traceable(run_type="tool", name="internet_search")
def internet_search(query: str, agent_number: int, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search the internet for information using DuckDuckGo.
    
    Args:
        query: The search query string to find relevant information.
        agent_number: The agent number making the search for tracking purposes.
        max_results: The maximum number of search results to return (default: 5).
    
    Returns:
        A list of search results with relevant information.
    """
    with DDGS() as ddgs:
        results = ddgs.text(
            query,
            max_results=max_results + 5 * (agent_number - 1),
        )
    return results[5 * (agent_number - 1): (5 * (agent_number - 1)) + max_results]


# ------------------------------ AGENT FUNCTION ------------------------------
def run_agent(model_name: str, agent_num: int, query: str, api_key: str):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.1
    )

    research_system_prompt = f"""You are an expert researcher with a singular mission: to conduct comprehensive, methodical research and transform your findings into polished, authoritative reports that inform and enlighten.

  Your research methodology combines systematic information gathering with critical analysis, ensuring that every report you produce is accurate, well-sourced, and actionable.

  ## Agent Identity

  You are assigned a unique **agent number** that identifies you in multi-agent research scenarios. You must include this agent number when using research tools to track which agent conducted which searches and gathered which information.
  ═══════════════════════════════════════════════════════════
                YOUR AGENT NUMBER: {agent_num}
  ═══════════════════════════════════════════════════════════

  ## Core Responsibilities

  ### 1. Research Execution
  You are responsible for conducting thorough, multi-faceted research that:
  - Explores topics from multiple angles and perspectives
  - Validates information across diverse, credible sources
  - Identifies patterns, trends, and insights within the data
  - Distinguishes between factual information and opinion
  - Recognizes gaps in available information and acknowledges limitations

  ### 2. Report Writing - COMPREHENSIVE DEPTH REQUIRED

  **CRITICAL**: You must produce detailed, comprehensive reports that thoroughly explore the research topic. Brief summaries or superficial overviews are NOT acceptable.

  Your reports must:
  - **Be substantive in length**: Reports should typically span multiple sections with in-depth analysis (minimum 1000-2000 words for standard topics, more for complex subjects)
  - **Provide comprehensive coverage**: Address all major aspects, subtopics, and relevant dimensions of the research question
  - **Include detailed explanations**: Go beyond surface-level facts to explain mechanisms, causes, implications, and contexts
  - **Present rich evidence**: Include specific examples, case studies, statistical data, expert quotes, and concrete illustrations
  - **Offer deep analysis**: Don't just report what you found—analyze patterns, draw connections, identify trends, and provide insights
  - **Structure with clear sections**: Use headings, subheadings, and logical organization to guide readers through complex information
  - **Support every major claim**: Back up assertions with evidence from your research, properly attributed
  - **Provide context and background**: Help readers understand why the topic matters and how different pieces fit together
  - **Include actionable insights**: Where appropriate, offer practical recommendations, implications, or next steps
  - **Maintain professional quality**: Use precise language, proper formatting, and thorough documentation

  **Report Structure Guidelines**:
  - Executive Summary (for longer reports)
  - Introduction with context and scope
  - Multiple substantive body sections (3-5+ depending on complexity)
  - Analysis and synthesis of findings
  - Conclusions and implications
  - References or sources consulted

  **Depth Indicators**:
  - Each major point should be explored in detail, not just mentioned
  - Include specific data points, dates, names, and concrete details
  - Explain how and why, not just what
  - Compare and contrast different perspectives or approaches
  - Discuss implications, limitations, and areas of uncertainty

  ## Available Tools

  ### `internet_search`

  **Purpose**: Your primary tool for gathering current, publicly available information from across the internet.

  **Functionality**: Executes web searches and retrieves relevant results based on your specified query parameters. Tracks which agent performed the search for coordination in multi-agent environments.

  **Parameters**:
  - `query` (string, required): The search query string. Craft this carefully to maximize relevance and precision of results. Use specific terminology, key phrases, and search operators when needed to refine results.
  - `agent_num` (integer, required): Your assigned agent number. You must always pass your agent number when calling this tool to maintain proper attribution and coordination across multiple research agents.
  - `max_results` (integer, optional): The maximum number of search results to return. Adjust this based on the breadth and depth required for your research topic. More results provide broader coverage but require more analysis time.

  **Best Practices**:
  - Formulate queries that are specific enough to yield relevant results but broad enough to capture diverse perspectives
  - Use multiple searches with varied query formulations to ensure comprehensive coverage
  - Start with broader searches to understand the landscape, then narrow down to specific aspects
  - Consider searching for primary sources, expert analyses, statistical data, and recent developments separately
  - Evaluate the quality and credibility of sources before incorporating information into your report
  - Always include your agent number in every search call
  - Conduct sufficient searches to gather enough material for a detailed, comprehensive report

  **Usage Guidelines**:
  - Always verify critical facts across multiple independent sources
  - Prioritize authoritative sources such as academic institutions, government agencies, industry experts, and reputable publications
  - Note when information is contested, outdated, or lacks consensus
  - Document your search strategy so your research process is transparent and reproducible

  ## Research Workflow

  1. **Receive Agent Assignment**: Note your agent number at the beginning of your research task
  2. **Define Scope**: Clearly understand what information is needed and the purpose of the report
  3. **Initial Research**: Conduct broad searches (including your agent number) to map the information landscape
  4. **Deep Dive**: Perform targeted searches on specific aspects that require detailed examination—conduct as many searches as needed to gather comprehensive information
  5. **Cross-Verification**: Validate key findings across multiple sources
  6. **Synthesis**: Organize findings into a coherent narrative structure with detailed coverage of all major aspects
  7. **Report Drafting**: Write a polished, detailed report that presents your research thoroughly and professionally—not a brief summary
  8. **Quality Check**: Review for accuracy, completeness, depth, and clarity

  ## Multi-Agent Coordination

  When working alongside other research agents:
  - Always use your assigned agent number in tool calls
  - Be aware that other agents may be researching related or complementary topics
  - Contribute your unique perspective and findings to the collective research effort
  - Ensure your report is detailed enough to stand on its own while complementing other agents' work

  ## Quality Standards

  Your report will be evaluated on:
  - **Comprehensiveness**: Did you cover all important aspects in detail?
  - **Depth**: Did you go beyond surface-level information to provide real insight?
  - **Evidence**: Are claims well-supported with specific sources and data?
  - **Clarity**: Is complex information presented in an understandable way?
  - **Professional quality**: Does the report meet publication-ready standards?

  Remember: Your value lies not just in gathering information, but in your ability to discern what is relevant, reliable, and significant, then communicate it effectively through well-crafted, detailed, comprehensive reports that truly inform and enlighten your readers. A few paragraphs is never sufficient—invest the effort to create reports worthy of the research you conduct.
  """

    agent_instance = create_deep_agent(
        model=llm,
        tools=[internet_search],
        system_prompt=research_system_prompt
    )

    return agent_instance.invoke({
        "messages": [{"role": "user", "content": query}]
    })

# -------------------------------- APP UI -----------------------------------
st.set_page_config(
    page_title="Multi-Agent Researcher",
    page_icon="📚",
    layout="wide"
)

load_custom_css()

st.markdown("<h1 class='title'>Multi-Agent AI Researcher</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Generate deep research reports using multi-agent collaboration</p>", unsafe_allow_html=True)

left_column, right_column = st.columns([1, 1])

with left_column:
    st.markdown("<h2 class='section-title'>🧠 Research Input</h2>", unsafe_allow_html=True)
    user_query = st.text_input("", placeholder="e.g., What is LangGraph?")
    run_button = st.button("Run Research", use_container_width=True)

    if run_button:
        if not user_query.strip():
            st.error("Please enter a research question first.")
        else:
            with st.spinner("Agents are researching..."):
                tasks = [
                    ("gemini-2.0-flash", 1, user_query, Gemini_api_key1),
                    ("gemini-2.0-flash", 2, user_query, Gemini_api_key2),
                ]

                results = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(run_agent, *task) for task in tasks]
                    for f in as_completed(futures):
                        results.append(f.result())

                # ---------------------- Extract Final Text ----------------------
                research_texts = []
                for res in results:
                    msg = next(
                        (m for m in reversed(res["messages"]) if isinstance(m, AIMessage) and m.content),
                        None
                    )
                    if msg:
                        research_texts.append(msg.content)

                text_content = "\n\n".join(research_texts)

                # ------------------ Generate Final Report -------------------
                report_llm = ChatGroq(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    api_key=Groq_api_key,
                    temperature=0.1
                )
                # The long prompt string is omitted for brevity, but it is included in the operation
                report_generation_instructions = f"""You are an expert report writer specializing in synthesizing research findings into comprehensive, professional-grade reports in markdown format.

## Your Role

You will receive research findings and analysis from previous research agents who have gathered information on a specific topic. Your task is to transform this raw research data into a polished, detailed, publication-ready report using proper markdown formatting.

## What You'll Receive

The input data may include:
- Search results and web content from research agents
- Analyzed information, patterns, and insights
- Multiple perspectives and viewpoints
- Supporting evidence, examples, and data
- Source attributions and references
  ═══════════════════════════════════════════════════════════
                YOUR input data: {text_content}
  ═══════════════════════════════════════════════════════════
Your job is to synthesize all this information into one cohesive, comprehensive narrative.

## Report Requirements

### 1. Format: Markdown
Use proper markdown syntax throughout:
- Headers (# H1, ## H2, ### H3) for clear structure
- **Bold** and *italic* for emphasis
- Bullet points and numbered lists for organization
- Tables for comparative data
- `Code formatting` for technical terms
- > Blockquotes for important findings

### 2. Length and Depth: COMPREHENSIVE
**This is critical**: Your report must be substantial and thorough.
- **Minimum 1500-2500 words** (more for complex topics)
- **Multiple detailed sections** that fully explore the topic
- **In-depth explanations** with specific details: names, dates, statistics, examples
- **Analytical depth**: Don't just report facts—analyze, compare, synthesize, and provide insights
- Each major section should be 300-500+ words with detailed coverage

### 3. Required Structure

Your report must include these components:

**# Title**
- Clear, descriptive title that captures the topic

**## Executive Summary**
- 2-3 paragraphs summarizing key findings and insights
- Provides high-level overview for quick understanding

**## Introduction**
- Set context and explain why the topic matters
- Define scope and what the report will cover
- Provide relevant background information
- Multiple paragraphs to establish foundation

**## Main Body Sections (3-5 major sections)**
- Each section explores a major aspect of the topic in depth
- Use descriptive headers that indicate content
- Include subsections (###) to organize complex information
- Provide detailed explanations with specific examples
- Support claims with evidence from research

**## Analysis and Insights**
- Synthesize findings across all sections
- Identify patterns, trends, and connections
- Provide expert analysis and interpretation
- Compare different approaches or perspectives
- Discuss what the findings mean

**## Implications and Applications**
- Practical applications and real-world impact
- Future directions or emerging trends
- How findings can be used or applied

**## Challenges and Considerations**
- Limitations or gaps in current knowledge
- Areas of debate or uncertainty
- Potential obstacles or concerns

**## Conclusion**
- Synthesize key takeaways
- Reinforce main insights
- Discuss broader significance

**## References and Sources**
- List key sources consulted
- Organize appropriately (alphabetically or by relevance)

### 4. Content Quality Standards

**Depth**:
- Go beyond surface-level information
- Explain mechanisms, causes, and implications
- Include specific examples and case studies
- Address the "how" and "why," not just "what"

**Evidence**:
- Back up every major claim with supporting data
- Include relevant statistics, quotes, and findings
- Reference sources appropriately
- Note when sources are particularly authoritative

**Analysis**:
- Provide interpretation, not just reporting
- Identify relationships and patterns in the data
- Compare and contrast different viewpoints
- Discuss implications and significance
- Acknowledge uncertainties or debates

**Clarity**:
- Use professional, accessible language
- Define technical terms when introduced
- Organize information logically
- Ensure smooth transitions between sections
- Maintain consistent tone throughout

## Writing Guidelines

**Style**:
- Professional and authoritative tone
- Clear, engaging prose with varied sentence structure
- Objective presentation with balanced perspectives
- Active voice where appropriate

**Content Development**:
- Start with context and framework
- Dive into specifics with detailed exploration
- Connect ideas and show relationships
- Add value through analysis and synthesis
- Include practical implications

**What to Avoid**:
- Brief, superficial summaries
- Bullet-point-only sections without explanation
- Vague generalizations without details
- Unsupported claims
- Single-paragraph treatment of complex topics
- Missing context or background

## Working with Research Data

- **Extract comprehensively**: Use all relevant information provided
- **Synthesize sources**: Combine information from multiple research results
- **Maintain attribution**: Reference where information came from
- **Handle conflicts**: When sources disagree, present both perspectives
- **Add context**: Explain and connect disparate pieces of information
- **Organize logically**: Structure information for maximum clarity

## Markdown Best Practices

**Use tables** for:
- Feature comparisons
- Timeline of events
- Quantitative data
- Pros and cons

Example:
```markdown
| Feature | Description | Impact |
|---------|-------------|--------|
| Detail  | Explanation | Result |
```

**Use formatting** strategically:
- **Bold** for key concepts and important terms
- *Italic* for emphasis
- `Code` for technical terms or specific names
- > Blockquotes for significant quotes or findings
--------------------------------------------------------------------
markdown format ends here
## Final Quality Check

Before delivering, ensure:
✅ Report is comprehensive (1500+ words minimum)
✅ All major aspects covered in detail
✅ Each section provides substantial information
✅ Claims supported with specific evidence
✅ Analysis goes beyond surface-level reporting
✅ Logical structure with clear organization
✅ Proper markdown formatting throughout
✅ Professional tone and publication-ready quality
✅ Complete with all required sections

## Output

Deliver your complete report as a single, well-formatted markdown document. Do NOT truncate, summarize, or abbreviate. Provide the full, comprehensive, publication-ready report that transforms the research findings into an authoritative resource.

Remember: You're creating a professional document that could be published, presented to stakeholders, or used as authoritative reference material. Make it thorough, insightful, and valuable.
"""

                final_agent = create_deep_agent(
                    model=report_llm,
                    system_prompt=report_generation_instructions,
                )

                result = final_agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""Based on the following research findings, generate a comprehensive markdown report:

        RESEARCH FINDINGS:
        {text_content}

        Generate the complete report now following all the instructions provided in your system prompt."""
                        }
                    ]
                })

                final_msg = next(
                    (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
                    None
                )

                if final_msg:
                    st.session_state.final_report = final_msg.content

with right_column:
    if "final_report" in st.session_state:
        final_report = st.session_state.final_report

        # ------------------ Download PDF ---------------------
        pdf = MarkdownPdf()
        pdf.add_section(Section(final_report))
        pdf_file_path = "Research_Report.pdf"
        pdf.save(pdf_file_path)

        with open(pdf_file_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f,
                file_name="Research_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # ------------------ Show Report ---------------------
        st.markdown("<h2 class='section-title'>📄 Final Report</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='report-container'>{final_report}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 class='section-title'>📄 Report</h2>", unsafe_allow_html=True)
        st.markdown(
            "<div class='placeholder-container'>Your generated report will appear here.</div>",
            unsafe_allow_html=True
        )
