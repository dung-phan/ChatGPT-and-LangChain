import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

langchain.debug = True

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv('AZURE_ENDPOINT'),
    deployment=os.getenv('AZURE_DEPLOYMENT_NAME'),
    api_key=os.getenv('AZURE_API_KEY'),
)
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
) # Initialize the Chroma vector store to query the database

llm = OpenAI(model="gpt-4o-mini")
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    db=db,
)
prompt = PromptTemplate.from_template(
    "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
)
qa_chain = ({
    "context": retriever,
    "question": RunnablePassthrough(),
}
    | prompt
    | llm
)
result = qa_chain.invoke("What is an interesting fact about the English language?")
print(result)
