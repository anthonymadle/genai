from fastapi import FastAPI

app = FastAPI(title="Class Project API")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with UV!"}
