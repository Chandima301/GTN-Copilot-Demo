from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.chains import ConversationChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains.router import MultiPromptChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All

def get_task_automation_chain():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # local_path = (
    #     "./models/mistral-7b-instruct-v0.1.Q4_0.gguf"  # replace with your desired local file path
    # )
    # callbacks = [StreamingStdOutCallbackHandler()]
    # llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

    # Schema
    order_schema = {
        "properties": {
            "automation_request_type": {"enum": ["order", "user_details"]},
            "symbol": {"enum": ["APPL", "TSLA", "GOOGL", "NFLX", "ADBE", "ADI", "KDP"]},
            "order_side": {"enum": ["buy", "sell"]},
            "order_value": {"type": "number"},
            "quantity": {"type": "integer"},
            "order_type": {"type": "string"},
        },
        "required": ["symbol", "order_side"],
    }

    # Schema
    fill_form_schema = {
        "properties": {
            "automation_request_type": {"enum": ["order", "user_details"]},
            "name": {"type" : "string"},
            "age": {"type" : "string"},
            "address": {"type" : "string"}
        },
        "required": ["name", "age"],
    }



    chain_schemas = [
        {
            "name": "order",
            "description": "Use for placing buy and sell orders",
            "schema": order_schema,
        },
        {
            "name": "user",
            "description": "Use for collecting user details",
            "schema": fill_form_schema,
        },
    ]

    destination_chains = {}
    
    for p_info in chain_schemas:
        name = p_info["name"]
        schema = p_info["schema"]
        chain = create_extraction_chain(schema, llm)
        destination_chains[name] = chain
    
    default_chain = ConversationChain(llm=llm, output_key="text")

    destinations = [f"{p['name']}: {p['description']}" for p in chain_schemas]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )

    return chain