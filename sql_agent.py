import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
llm=ChatGroq(groq_api_key=api_key,model_name="gemma2-9b-it",streaming=True)

#db_uri = "postgresql://postgres:1234@localhost:5432/hr"
postgresql_host = "localhost"
postgresql_user = "postgres"
postgresql_password = "1234"
postgresql_db = "hr"


def config_db(postgresql_host,postgresql_user,postgresql_password,postgresql_db):
        return SQLDatabase(create_engine(f"postgresql+psycopg2://{postgresql_user}:{postgresql_password}@{postgresql_host}/{postgresql_db}"))

# print(f"Database created successfully! {df}")

# Part 2: Prepare the sql prompt
SQL_AGENT_PREFIX = """

You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query
to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the
below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the
tools below.
- as part of your final answer, please include the SQL query you used in json format or code format

## Tools:

"""

SQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action Input: 
SELECT TOP (10) [base_salary], [grade] 
FROM salaries_2023

WHERE state = 'Division'

Observation:
[(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Thought:I now know the final answer
Final Answer: There were 27437 workers making 100,000.

Explanation:
I queried the `xyz` table for the `salary` column where the department
is 'IGM' and the date starts with '2020'. The query returned a list of tuples
with the bazse salary for each day in 2020. To answer the question,
I took the sum of all the salaries in the list, which is 27437.
I used the following query

```sql
SELECT [salary] FROM xyztable WHERE department = 'IGM' AND date LIKE '2020%'"
```
===> End of Example

"""



db=config_db(postgresql_host,postgresql_user,postgresql_password,postgresql_db)
toolkit=SQLDatabaseToolkit(db=db,llm=llm)

QUESTION = """what is the highest average salary by department, and give me the number?"
"""
sql_agent = create_sql_agent(
    prefix=SQL_AGENT_PREFIX,
    format_instructions=SQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=llm,
    toolkit=toolkit,
    top_k=30,
    verbose=True,
)

# res = sql_agent.invoke(QUESTION)

# print(res)

import streamlit as st

st.title("SQL Query AI Agent")

question = st.text_input("Enter your query:")

if st.button("Run Query"):
    if question:
        res = sql_agent.invoke(question)

        st.markdown(res["output"])
else:
    st.error("Please enter a query.")