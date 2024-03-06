from dotenv import load_dotenv
import os 
import streamlit as st
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine 
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from pdf import combined_engine
from pdf import get_index as pdf_get_index
load_dotenv()

population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
    )
population_query_engine.update_prompts({"pandas_prompt" : new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_query_engine",
            description="This gives information at the world population and demographic",
        ),
    ),
    # QueryEngineTool(
    #     query_engine=malaysia_engine,
    #     metadata=ToolMetadata(
    #         name="malaysia_data",
    #         description="This gives details information about Malaysia country",
    #     ),
    # ),
        QueryEngineTool(
        query_engine=combined_engine,
        metadata=ToolMetadata(
            name="combined_data",
            description="This gives information from multiple files",
        ),
    ),
]


llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

# while (prompt := input("Enter a prompt (q to quit): ")) != "q":
#     result = agent.query(prompt)
#     print(result)

file_paths = []

# File uploader in the sidebar
with st.sidebar:
    st.header("Upload PDF file")
    file = st.file_uploader("", type=["pdf"])

    if file:
        file_path = os.path.join("data", file.name)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        
        # Display confirmation message
        st.success(f"File uploaded successfully: {file.name}")

        # Add the uploaded file path to the list of file paths
        file_paths.append(file_path)

# Check if there are any uploaded files
if file_paths:
    # Get combined index for uploaded files
    combined_index = pdf_get_index(file_paths, 'combined_index')
    combined_engine = combined_index.as_query_engine()

    # Add QueryEngineTool for combined data
    tools.append(
        QueryEngineTool(
            query_engine=combined_engine,
            metadata=ToolMetadata(
                name="combined_data",
                description="This gives information from multiple files",
            ),
        )
    )


st.title("AgentAI - RAG")

user_input = st.text_input("Enter a prompt:")
if user_input:
    if user_input.lower() == 'q':
        st.stop()
    else:
        result = agent.query(user_input)
        st.text_area("Response:", value=result, height=100, disabled=False)
    