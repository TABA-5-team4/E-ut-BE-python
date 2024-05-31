from fastapi import FastAPI, File
from typing import Union
from pydantic import BaseModel

from gptApi import chat_with_gpt, invoke_chain
from model.request import ChatRequest, STTRequest

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
def chat(chatRequest: ChatRequest):
    # response = chat_with_gpt(text)
    response = invoke_chain(chatRequest.text)
    return {"response": response}

@app.post("/stt")
def stt(voiceFile: bytes = File(...)):
    # todo
    return {
    "audio_length": 7.983457,
    "stt_result": "우리 집에 잘 못 산다는 것을 친구들이 알게 되었을 때 정말 억장이 무너지는 거 같았어",
    "gpt_response": "그렇게 슬픈 생각을 하고 계시네요. 혼자 계시다 보니 더 그런 마음이 들 수 있죠. 언제든지 제가 옆에 있어요. 함께 이야기를 나누면 마음이 편안해질 거에요.",
    "sentiment_analysis": [
        {
            "label": "슬픔",
            "score": 0.8891498446464539
        },
        {
            "label": "분노",
            "score": 0.08132488280534744
        },
        {
            "label": "당황",
            "score": 0.01866663619875908
        },
        {
            "label": "불안",
            "score": 0.008278260938823223
        },
        {
            "label": "행복",
            "score": 0.0009516564896330237
        },
        {
            "label": "중립",
            "score": 0.0009417913970537484
        },
        {
            "label": "혐오",
            "score": 0.0006868354394100606
        }
    ]
}