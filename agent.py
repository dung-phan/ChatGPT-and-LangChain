from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from tools.report import write_report_tool
from tools.sql import run_query_tool, list_tables, describe_tables_tool

load_dotenv()
llm = ChatOpenAI(
    model="gpt-4o-mini",
)

tables = list_tables()

prompt = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessage(
            content=(
                "You are an AI that has access to a SQLite database.\n"
                f"The database has tables of: {list_tables()}\n"
                "Do not make any assumptions about the data in the tables.\n"
                "Instead, use the tools provided to answer questions about the data.\n"
            ),
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [run_query_tool, describe_tables_tool, write_report_tool]
agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)
agent_executor.invoke({
    "input": "Summarize the top 5 most popular product. Write the results to a report file."
})
