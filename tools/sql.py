import sqlite3
from typing import List

from langchain_core.tools import Tool
from pydantic.v1 import BaseModel

connection = sqlite3.connect("db.sqlite")

def list_tables():
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    tables = cursor.fetchall()
    return "\n".join([table[0] for table in tables])

def run_sqlite_query(query: str):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.OperationalError as e:
        return f"The following error occurred: {str(e)}"

class RunQueryArgsSchema(BaseModel):
    query: str

run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a SQL query on the SQLite database",
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema,
)

def describe_tables(table_names):
    cursor = connection.cursor()
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = cursor.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table'and name IN ({tables});"
    )
    return '\n'.join([row[0] for row in rows])

class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]

describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, return the SQL schema for each table.",
    func=describe_tables,
    args_schema=DescribeTablesArgsSchema
)
