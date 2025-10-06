# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# ----- Bigram imports -----
from app import bigram_model
from app.embedding import get_embedding

# ----- CNN imports -----
import torch
from io import BytesIO
from PIL import Image
from torchvision import transforms
from app.cnn_model import SimpleCNN

app = FastAPI(title="GenAI Class API")

# =========================
# Bigram endpoints
# =========================
# Build a tiny corpus ONCE (no printing)
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
all_text = " ".join(corpus)
vocab, bigram_probs = bigram_model.analyze_bigrams(all_text)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = 20

@app.get("/")
def read_root():
    return {"status": "ok", "message": "GenAI Class API running"}

@app.post("/generate")
def generate(request: TextGenerationRequest):
    try:
        generated_text = bigram_model.generate_text(
            bigram_probs, request.start_word, request.length
        )
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/embed")
def embed_text(text: str):
    try:
        embedding = get_embedding(text)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# =========================
# CNN classifier endpoint
# =========================
# Load the trained model ONCE at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = SimpleCNN().to(device)
cnn_model.load_state_dict(torch.load("models/cifar10_cnn.pt", map_location=device))
cnn_model.eval()

# CIFAR-10 labels
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Must match training transforms
_cnn_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

@app.post("/cnn/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        x = _cnn_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = cnn_model(x)
            pred = logits.argmax(dim=1).item()
        return {"predicted_class": CIFAR10_CLASSES[pred]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
