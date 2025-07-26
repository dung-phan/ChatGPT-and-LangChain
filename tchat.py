from langchain_openai import ChatOpenAI
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenAI(
    model="gpt-4o-mini",
)

prompt = ChatPromptTemplate.from_messages(
    messages=[
        HumanMessagePromptTemplate.from_template("{content}"),
    ]
)

chain = prompt | chat

while True:
    content = input(">> ")

    result = chain.invoke({
        "content": content,
    })
    print(result.content)
