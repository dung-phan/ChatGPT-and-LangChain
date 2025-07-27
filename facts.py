import os

from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
)
loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv('AZURE_ENDPOINT'),
    deployment=os.getenv('AZURE_DEPLOYMENT_NAME'),
    api_key=os.getenv('AZURE_API_KEY'),
)

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db"  # Specify the directory to persist the database
)

results = db.similarity_search_with_score("What is an interesting fact about the English language?",
                                          k=3)

for result, score in results:
    print(f"Score: {score}\nContent: {result.page_content}\nMetadata: {result.metadata}\n")
