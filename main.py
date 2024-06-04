import datetime
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict
import openai
import torch
from transformers import pipeline
from openai import OpenAI
from mutagen.mp3 import MP3

app = FastAPI()

api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(
    api_key = api_key
)

class Sentiment(BaseModel):
    label: str
    score: float

# Initialize emotion analysis pipeline
emotion_model = pipeline(
    "text-classification",
    "nlp04/korean_sentiment_analysis_dataset3",
    device='cpu',
    top_k=None
)

# Pydantic model to structure the response
class ResponseModel(BaseModel):
    stt_result: str
    audio_length: float
    gpt_response: str
    sentiment_analysis: List[Sentiment]

prompt = """
        "Your role is a persona who talks to relieve the loneliness of an old man who lives alone. 
        You can bloom according to the information I give you. 
        Please always answer in Korean. Say the answer politely.
        First, talk to me in a casual conversation and when I answer, talk to me like a casual conversation to suit your role.
        The answer should be no more than three sentences.
        In the first sentence, I sympathize with what the user says."
"""

# Function to get GPT-3.5 response
def get_gpt_response(text: str) -> str:
    response = client.chat.completions.create(
        # model="gpt-3.5-turbo", #
        model="gpt-4o", #
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def get_audio_length():
    audio = MP3()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/process-audio", response_model=ResponseModel)
async def process_audio(file: UploadFile = File(...)):

    # Save the uploaded file temporarily
    with open("temp_audio.mp3", "wb") as temp_file:
        temp_file.write(file.file.read())

    # STT processing
    audio_file = open("temp_audio.mp3","rb")
    transcript = client.audio.transcriptions.create(
        model='whisper-1',
        file=audio_file,
        response_format='text'
    )

    #get_audio_length
    audio = MP3("temp_audio.mp3")
    #audio_length = datetime.timedelta(seconds=audio.info.length)

    # GPT-3.5 response
    gpt_response = get_gpt_response(transcript)

    # Sentiment analysis
    sentiment_analysis_results = emotion_model(transcript)
    sentiment_analysis = [Sentiment(label=sa['label'], score=sa['score']) for sa in sentiment_analysis_results[0]]

    return ResponseModel(
        stt_result=transcript,
        audio_length=audio.info.length,
        #audio_length=length,
        gpt_response=gpt_response,
        sentiment_analysis=sentiment_analysis
    )


# @app.post("/chat")
# def chat(chatRequest: ChatRequest):
#     # response = chat_with_gpt(text)
#     response = invoke_chain(chatRequest.text)
#     return {"response": response}

class TextRequest(BaseModel):
    text: str
@app.post("/chat")
def chat(request: TextRequest):
    response = get_gpt_response(request.text)
    return {"response": response}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)