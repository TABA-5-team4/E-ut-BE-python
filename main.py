import os
import numpy as np
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

app = FastAPI()
# 데이터 압축해서 전송
app.add_middleware(GZipMiddleware, minimum_size = 500)

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = api_key
)

class Sentiment(BaseModel):
    label: str
    score: float

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
        Get rid of the correspondence."
"""

# Function to get GPT-4o response
def get_gpt_response(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

# Initialize emotion analysis pipeline
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, attention_mask, segment_ids):
        outputs = self.bert(input_ids=token_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            pooler_output = outputs[1]
        else:
            return None
        if self.dr_rate:
            pooler_output = self.dropout(pooler_output)
        return self.classifier(pooler_output)

# Define BERTDataset class
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        self.sentences = [
            bert_tokenizer(i[sent_idx], padding='max_length', max_length=max_len, truncation=True, return_tensors="pt")
            for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        input_ids = self.sentences[i]["input_ids"]
        attention_mask = self.sentences[i]["attention_mask"]
        token_type_ids = torch.zeros_like(input_ids)
        input_ids = input_ids.squeeze(0)
        token_type_ids = token_type_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        return input_ids, attention_mask, token_type_ids, self.labels[i]

    def __len__(self):
        return len(self.labels)

# Define prediction function
def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, max_len=64, pad=True, pair=False)
    test_dataloader = DataLoader(another_test, batch_size=1, num_workers=0)

    model.eval()
    for batch_id, (token_ids, attention_mask, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)

        with torch.no_grad():
            out = model(token_ids, attention_mask, segment_ids)

        for i in out:
            logits = i
            probabilities = F.softmax(logits, dim=0).detach().cpu().numpy()
            class_labels = ["당황", "불안", "분노", "슬픔", "중립", "행복", "혐오"]
            sorted_indices = np.argsort(probabilities)[::-1]
            sorted_probabilities = probabilities[sorted_indices]
            sorted_labels = [class_labels[idx] for idx in sorted_indices]

            results = [{"label": label, "score": float(score)} for label, score in zip(sorted_labels, sorted_probabilities)]
            return results

# Load model and tokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = AutoModel.from_pretrained('skt/kobert-base-v1', return_dict=False)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
# finetuning 모델 가중치 가져오기
model.load_state_dict(torch.load("./sentiment_analysis.pt", map_location=device))
model.eval()

# Summarize model
model_path = "./summary_model"
sm_model = BartForConditionalGeneration.from_pretrained(model_path)
sm_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

def get_summary(text):
    raw_input_ids = sm_tokenizer.encode(text)
    input_ids = [sm_tokenizer.bos_token_id] + raw_input_ids + [sm_tokenizer.eos_token_id]

    summary_ids = sm_model.generate(torch.tensor([input_ids]), max_length=100, min_length=10)
    summary_text = sm_tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    return summary_text

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/process-audio", response_model=ResponseModel)
async def process_audio(file: UploadFile = File(...)):

    # Save the uploaded file temporarily
    with open("temp_audio.mp4", "wb") as temp_file:
        temp_file.write(file.file.read())

    # STT processing
    audio_file = open("temp_audio.mp4","rb")
    transcript = client.audio.transcriptions.create(
        model='whisper-1',
        file=audio_file,
        response_format='text'
    )

    # Get audio length
    audio = MP3("temp_audio.mp4")

    # GPT-3.5 response
    gpt_response = get_gpt_response(transcript)

    # User + Gpt response -> summary
    summary_input = transcript + gpt_response
    summary_result = get_summary(summary_input)

    # Sentiment analysis
    sentiment_analysis_results = predict(transcript)
    sentiment_analysis = [Sentiment(label=sa['label'], score=sa['score']) for sa in sentiment_analysis_results]

    print(audio.info.length)

    return ResponseModel(
        stt_result=transcript,
        audio_length=audio.info.length,
        gpt_response=gpt_response,
        sentiment_analysis=sentiment_analysis,
        summary_result=summary_result
    )

class TextRequest(BaseModel):
    text: str

@app.post("/chat")
def chat(request: TextRequest):
    text = request.text + '와 관련된 안부를 물어봐주세요.'
    response = get_gpt_response(text)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)