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
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

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

# Summarize model
model_path = "C:/Users/Admin/Downloads/summary_model/saved_model"

sm_model = BartForConditionalGeneration.from_pretrained(model_path)
sm_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

def get_summary(text):
    raw_input_ids = sm_tokenizer.encode(text)
    input_ids = [sm_tokenizer.bos_token_id] + raw_input_ids + [sm_tokenizer.eos_token_id]

    summary_ids = sm_model.generate(torch.tensor([input_ids]), max_length=100, min_length=10)
    summary_text = sm_tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    return summary_text

# Pydantic model to structure the response
class ResponseModel(BaseModel):
    stt_result: str
    audio_length: float
    gpt_response: str
    sentiment_analysis: List[Sentiment]
    summary_result: str

prompt = """
        "Your role is a persona who talks to relieve the loneliness of an old man who lives alone. 
        You can bloom according to the information I give you. 
        Please always answer in Korean. Say the answer politely.
        First, talk to me in a casual conversation and when I answer, talk to me like a casual conversation to suit your role.
        The answer should be no more than three sentences.
        In the first sentence, I sympathize with what the user says.
        Get rid of the correspondence.
"
"""

# Function to get GPT-3.5 response
def get_gpt_response(text: str) -> str:
    response = client.chat.completions.create(
        # model="gpt-3.5-turbo", #
        model="gpt-4o", # 성능 넘사벽
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

    # User + Gpt response -> summary
    summary_input = transcript + gpt_response
    summary_result = get_summary(summary_input)

    # Sentiment analysis
    sentiment_analysis_results = emotion_model(transcript)
    sentiment_analysis = [Sentiment(label=sa['label'], score=sa['score']) for sa in sentiment_analysis_results[0]]

    return ResponseModel(
        stt_result=transcript,
        audio_length=audio.info.length,
        gpt_response=gpt_response,
        sentiment_analysis=sentiment_analysis,
        summary_result=summary_result
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
    text = request.text + '와 관련된 안부를 물어봐주세요.'
    response = get_gpt_response(text)
    return {"response": response}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)