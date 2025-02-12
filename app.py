from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer
import os

app = FastAPI()

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Intentional bug: No token truncation for long texts
model = pipeline("text-classification", model=model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def chunk_text(text, max_length=512):
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    return [tokenizer.decode(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length)]

# OR use the truncate and padding built-in options.
# model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", 
#                  tokenizer="distilbert-base-uncased-finetuned-sst-2-english", truncation=True, padding=True)


@app.post("/predict")
async def predict(text: str):
    
    chunks = chunk_text(text, max_length=512)

    results = [model(chunk) for chunk in chunks]
    return results
