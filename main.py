from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel

from gptApi import chat_with_gpt, invoke_chain
from model.request import ChatRequest

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.post("/chat")
async def chat(chatRequest: ChatRequest):
    # response = chat_with_gpt(text)
    response = await invoke_chain(chatRequest.text)
    return {"response": response}