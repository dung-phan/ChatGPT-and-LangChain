from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenAI(
    model="gpt-4o-mini",
)

prompt = ChatPromptTemplate.from_messages(
    messages=[
        MessagesPlaceholder("messages"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)
store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = prompt | chat
chat_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="messages",
)

while True:
    content = input(">> ")

    result = chat_with_history.invoke(
        {"input": content},
        config={"configurable": {"session_id": "test_session"}}
    )
    print(result.content)
