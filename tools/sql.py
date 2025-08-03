import sqlite3

from langchain_core.tools import Tool

connection = sqlite3.connect("db.sqlite")

def run_sqlite_query(query: str):
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchall()

run_query_tool = Tool(
    name="run_sqlite_query",
    description="Run a SQL query on the SQLite database",
    func=run_sqlite_query,
)
