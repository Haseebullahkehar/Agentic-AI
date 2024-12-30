# import os
from phi.agent import Agent
from phi.model.groq import Groq
# from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# Load enviroment variables
load_dotenv()


web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    # model=OpenAIChat(id="gpt-3.5-turbo"),
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    # model=OpenAIChat(id="gpt-3.5-turbo"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,
                         company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team.print_response(
    "Summarize analyst recommendations and share the latest news about NVDA", stream=True)
