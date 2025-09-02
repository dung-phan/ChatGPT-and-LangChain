from langchain_core.tools import StructuredTool
from pydantic.v1 import BaseModel

class ReportArgsSchema(BaseModel):
    filename: str
    html: str


def write_report(filename, html):
    with open(filename, "w") as f:
        f.write(html)

write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write an HTML report to a file. Use this tool whenever someone asks you to write a report.",
    func=write_report,
    args_schema=ReportArgsSchema
)
