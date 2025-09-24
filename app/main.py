
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app import bigram_model
from app.embedding import get_embedding


app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

all_text = " ".join(corpus)
vocab, bigram_probs = bigram_model.analyze_bigrams(all_text)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(bigram_probs, request.start_word, request.length)
    return {"generated_text": generated_text}


@app.get("/embed")
def embed_text(text: str):
    embedding = get_embedding(text)
    return {"embedding": embedding}
