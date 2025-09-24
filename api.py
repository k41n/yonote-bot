from fastapi import FastAPI
from pydantic import BaseModel
from rag import rag

app = FastAPI()

# GET endpoints
@app.get("/")
def read_root(q: str):
  return rag.query(q)

