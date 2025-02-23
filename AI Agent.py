import os
import streamlit as st
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

# Load environment variables
load_dotenv()

# Retrieve API keys from the environment
deepseek_api_key = os.getenv("GROQ_DEEPSEEK_API_KEY")
qwen_api_key = os.getenv("GROQ_QWEN_API_KEY")

# Debugging API key loading
if not deepseek_api_key or not qwen_api_key:
    raise ValueError("Missing API keys. Check your .env file.")
print("DeepSeek API Key Loaded")
print("Qwen API Key Loaded")

# Define the Web Agent using Groq's QWEN model
web_agent = Agent(
    name="Web Agent",
    model=Groq(id="qwen-2.5-coder-32b", api_key=qwen_api_key),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Define the Finance Agent using Groq's DeepSeek model
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="qwen-2.5-coder-32b", api_key=qwen_api_key),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Combine agents into a team
agent_team = Agent(
    model=Groq(id="deepseek-r1-distill-llama-70b", api_key=deepseek_api_key),
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Debugging agent_team initialization
if not agent_team:
    raise ValueError("Agent team failed to initialize.")
print("Agent team initialized successfully")

# Streamlit UI
st.set_page_config(page_title="AI Chat Assistant", layout="wide")
st.title("🤖 AI Chat Assistant")

# User input
if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        response_text = ""

        # Ensure agent_team has a valid method to generate responses
        if hasattr(agent_team, "respond") and callable(agent_team.respond):
            try:
                response_text = web_agent.respond(prompt)  # Try this instead
                response_container.markdown(response_text if response_text else "No response received.")
            except Exception as e:
                error_message = f"Error during response: {str(e)}"
                print(error_message)
                response_container.markdown(error_message)
        else:
            error_message = "Error: Agent does not support valid response methods."
            print(error_message)
            response_container.markdown(error_message)
