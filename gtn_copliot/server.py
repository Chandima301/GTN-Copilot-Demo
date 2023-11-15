from fastapi import FastAPI
from langserve import add_routes
import os

from gtn_copliot.chain import get_graphql_chain, get_graphql_input_prompt, get_output_parser
from gtn_copliot.admin_assist_chain import get_admin_assist_chain
from gtn_copliot.task_automation_chain import get_task_automation_chain
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


add_routes(
    app,
    get_graphql_input_prompt() | get_graphql_chain(),
    path="/dynamic-data-retrieve",
)


add_routes(
    app,
    get_admin_assist_chain(),
    path="/admin-assist",
)

add_routes(
    app,
    get_task_automation_chain(),
    path="/task-automation",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT"))
