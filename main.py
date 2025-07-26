from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
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

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

chain = code_prompt | llm

result = chain.invoke({
    "language": args.language,
    "task": args.task
})
print(result)

