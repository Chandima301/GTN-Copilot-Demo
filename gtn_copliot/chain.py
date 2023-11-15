from typing import Optional

from langchain.llms import OpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import Runnable
from langchain.memory import ConversationBufferMemory

from langchain.schema import StrOutputParser

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain.chains import SimpleSequentialChain
from langchain.llms import GPT4All
from langchain.output_parsers import PydanticOutputParser

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.prompts import MessagesPlaceholder
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

from dotenv import load_dotenv

load_dotenv()


def create_agent(llm, llm_math):
    tools = load_tools(
        ["graphql", "llm-math"],
        graphql_endpoint="http://localhost:8806/graphql",
        llm=llm_math
    )

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
    )  

    return agent


# def create_agent(llm, llm_math):
    
#     tools = load_tools(
#         ["graphql", "llm-math"],
#         graphql_endpoint="http://localhost:8806/graphql",
#         llm=llm_math
#     )

def get_output_parser():
    response_schemas = [
            ResponseSchema(name="type", description="the type of the answer. if the answer contains just text, the type should be 'text'. If the answer containes multiple objects such as multiple account details, the type should be an 'array'"),
            ResponseSchema(name="content", description="The answer to the user request. The answer should be in text or array of objects. It should be text if the type is 'text'. it should be array of json objects, if the answer type is 'array'")
        ]
    
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return parser



def get_graphql_input_prompt():
    template = """
                Give data for the follwing user request,
                "{user_request}"

                Follow below instruction when retriving data
                
                *Use data that are stored in the graphql database having the follwing schema.
                *Always use deined queries only. 
                *Always use defined query parameters only.
                *Always request field names that are defined only in types.
                *Do not use made up field names, queries or query parameters.
                *Do not use aggregate queries.
                *Minimize the number of request to the graphql database.
                *Always use one query at a time.
                *Action Input should be always single line query.
                *Always use the simplest query unless the user asked for specific data.
                *Do not get account details by account numbers unless the user specifically asks.
                *Use only one intermediate step.
                
                *Strictly follow the below format instruction in the final answer
                {format_instructions}

                SCHEMA:
                """ + os.getenv("GRAPH_QL_SCHEMA")
    
    output_parser = get_output_parser()
    format_instructions = output_parser.get_format_instructions()
    return PromptTemplate(template=template, input_variables=["user_request"],partial_variables={"format_instructions": format_instructions})


def get_formatting_chain(llm):
    template = """
                Input: '{input}'
               
                Format the whole input into text or array and return as following output format,

                *Return only a json object having type field which specify the the type of the content as "text" or "array", also content field which gives a json object array or text.
                *A json object array should contain json objects of single level of fields.
                *If the input is in an array type or a list having multiple fields, it should be converted in to json objects. 

                Note: Final answer should contains some of the data retrived from the graphQL database.
                """
    prompt_template = PromptTemplate(input_variables=["input"], template=template)
    return LLMChain(llm=llm, prompt=prompt_template) 


def get_graphql_chain(model: Optional[BaseLanguageModel] = None) -> Runnable:
    llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo-16k", streaming=True, callbacks=[StreamingStdOutCallbackHandler()] )

    # local_path = (
    #     "./models/gpt4all-falcon-q4_0.gguf"  # replace with your desired local file path
    # )
    # callbacks = [StreamingStdOutCallbackHandler()]
    # llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
    llm_math = OpenAI(temperature=0)

    agent = create_agent(llm, llm_math)

    formatting_chain = get_formatting_chain(llm)
 #   return SimpleSequentialChain(chains=[agent, formatting_chain], verbose=True)
    return agent
