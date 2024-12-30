import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
from io import StringIO
import sys
import re

# Load environment variables
load_dotenv()

# Helper function to strip ANSI escape sequences
def strip_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x1B\x9B][0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# Web search agent
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=False,  # Disable tool call logs
    markdown=True,
)

# Finance agent
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=False,  # Disable tool call logs
    markdown=True,
)

# Team agent
agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=False,  # Disable tool call logs
    markdown=True,
)

# Streamlit UI
st.title("AI Agent Team for Web and Finance Data")

query = st.text_input("Enter your query:", "Summarize analyst recommendations and share the latest news for NVDA")

if st.button("Get Response"):
    st.write("Processing your request...")

    # Capture the output of print_response
    output_buffer = StringIO()
    sys.stdout = output_buffer  # Redirect stdout to capture the output

    # Use the print_response method
    agent_team.print_response(query, stream=False)

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Get the captured output and clean it
    raw_response = output_buffer.getvalue()
    clean_response = strip_ansi_escape_sequences(raw_response)

    # Display response
    if clean_response.strip():
        st.markdown("### Response:")
        st.markdown(clean_response)
    else:
        st.error("No response received. Please check the query or agent configuration.")
