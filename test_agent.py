from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """You are an agent that solves tasks using tools.

Use EXACTLY this loop format, one field per line:
Question: {input}
Thought: <your reasoning>
Action: <one of: {tool_names}>
Action Input: <args>
Observation: <tool result>
(repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: <answer>

{tools}

Begin!
Question: {input}
{agent_scratchpad}"""


@tool
def calculator(expression):
    """Do basic arithmetic like 13*17 or (2+3)/5"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

@tool
def current_time(_: str = ""):
    """Return the current time in ISO format"""
    return datetime.now().isoformat(timespec="seconds")

tools = [calculator, current_time]

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6)

print(agent_executor.invoke({"input": "What is 14*17, and also tell me the time in Amsterdam?"}))
