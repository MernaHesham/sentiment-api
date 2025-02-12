from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

from optimum.onnxruntime import ORTModelForSequenceClassification

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


## Quantized endpoint
## Fine-tuned approach

# fined_tuned_gpt2_name = "lvwerra/gpt2-imdb" 
# fined_tuned_gpt2 = AutoModelForSequenceClassification.from_pretrained(fined_tuned_gpt2_name)
# tokenizer_quantized = AutoTokenizer.from_pretrained(fined_tuned_gpt2)

# model_quantized = pipeline("text-classification", model=fined_tuned_gpt2, tokenizer=tokenizer_quantized)

# @app.post("/predict_quantized")
# async def predict(text: str):

#     results = model_quantized(text)
    
#     return results

## Use current BERT Quantized version

model_name_q = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
int8_model = ORTModelForSequenceClassification.from_pretrained(model_name_q)
model_q = pipeline("text-classification", model=int8_model, tokenizer=tokenizer)

@app.post("/predict_quantized")
async def predict(text: str):
    
    chunks = chunk_text(text, max_length=512)

    results = [model_q(chunk) for chunk in chunks]
    
    return results
