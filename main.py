from langchain_core.runnables import RunnableMap
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
import argparse
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()


llm = OpenAI(
    model="gpt-4o-mini",
)

code_prompt = ChatPromptTemplate.from_template("Write a very short {language} function that will {task}")

test_prompt = ChatPromptTemplate.from_template("Write a test for this {function}")

code_chain = code_prompt | llm
test_chain = test_prompt | llm

composed_chain = RunnableMap({
    "function": code_chain,
    "test": code_chain | (lambda fn: {"function": fn}) | test_chain
})

result = composed_chain.invoke({
    "language": args.language,
    "task": args.task,
})
print("Function:\n", result["function"])
print("\nTest:\n", result["test"])
