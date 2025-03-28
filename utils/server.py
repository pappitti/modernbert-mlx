# fastapi_bert_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import uvicorn

app = FastAPI(title="ModernBERT Inference API", 
              description="API for using ModerBERT",
              version="1.0.0")

# TOOD


# Define request and response models
class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    shape: int

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    shape: int

@app.post("/predict", response_model=EmbeddingResponse)
async def predict(request: TextRequest):
    """
    Get BERT embeddings for a single text
    """
    # Tokenize the input
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the [CLS] token embedding
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return {
        "embedding": embeddings.tolist()[0],
        "shape": embeddings.shape[1]
    }

@app.post("/batch_predict", response_model=BatchEmbeddingResponse)
async def batch_predict(request: BatchTextRequest):
    """
    Get BERT embeddings for multiple texts
    """
    if len(request.texts) > 32:
        raise HTTPException(status_code=400, detail="Batch size should not exceed 32")
    
    results = []
    for text in request.texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        results.append(embedding.tolist()[0])
    
    return {
        "embeddings": results,
        "shape": outputs.last_hidden_state.shape[2]
    }

@app.get("/health")
async def health_check():
    """
    Check if the server is healthy
    """
    return {"status": "healthy", "model": MODEL_NAME, "device": device.type}

if __name__ == "__main__":
    uvicorn.run("fastapi_bert_server:app", host="0.0.0.0", port=5000, workers=1)