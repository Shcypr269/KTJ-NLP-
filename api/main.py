from fastapi import FastAPI
from pydantic import BaseModel
from agent import AgenticRAGAssistant
from config import AgentConfig

app = FastAPI()

config = AgentConfig.from_env()
assistant = AgenticRAGAssistant(config)
assistant.initialize()

class ChatRequest(BaseModel):
    query: str

@app.post("/")
def chat(req: ChatRequest):
    return assistant.process_query(req.query)
