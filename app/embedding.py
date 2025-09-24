import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def get_embedding(text: str):
    doc = nlp(text)
    return doc.vector.tolist()
