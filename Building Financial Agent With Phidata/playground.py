import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
# from phi.storage.agent.sqlite import SqlAgentStorage
import phi.playground
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv

import os
import phi

# Load envriment variables from .env file
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-3.5-turbo"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    # storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIChat(id="gpt-3.5-turbo"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,
                         company_info=True, company_news=True)],
    instructions=["Use tables to display data"],
    # storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

app = Playground(agents=[finance_agent, web_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
