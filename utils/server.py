# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Union
import uvicorn
import gc
import mlx.core as mx
from .utils import PIPELINES, load

app = FastAPI(title="ModernBERT Inference API", 
              description="API for using ModerBERT",
              version="1.0.0")

# TOOD
# Model Unloading: Add an endpoint to unload specific models when they're not needed to save memory.
# Separate Services: For complete isolation, run each pipeline type as a separate FastAPI service and use a lightweight API gateway to route requests.
# Worker Pool Architecture: Implement a worker pool where each worker specializes in a specific pipeline, and a dispatcher routes requests to the appropriate worker.


# Model cache to avoid reloading
model_cache = {}

def get_model(model_name: str, pipeline_name: str):
    """
    Factory function to get or create the appropriate model
    """
    global model_cache

    if pipeline_name not in PIPELINES:
        raise HTTPException(status_code=400, detail=f"Pipeline '{pipeline_name}' not found. Available pipelines: {PIPELINES}")
    
    # Return from cache if already loaded
    if model_cache.get("pipeline", None)==pipeline_name and model_cache.get("model_name", None)==model_name:
        return model_cache
    
    # Load the appropriate model based on configuration
    if model_cache:
        # Clear references to existing model and tokenizer
        model_cache = {}
        
        # Force garbage collection
        gc.collect()
    
    model, tokenizer = load(
        model_name,
        pipeline=pipeline_name
    ) 

    # Update the cache
    model_cache = {
        "model_name": model_name,
        "pipeline": pipeline_name,
        "model": model,
        "tokenizer": tokenizer,
    }
    
    return model_cache

# Define request and response models
class PredictionRequest(BaseModel):
    text: Union[str, List[str]]
    model: str
    pipeline: str = "masked-lm"
    reference_text: Optional[Union[str, List[str]]] = None
    label_candidates: Optional[Union[dict, List[str]]] = None

class PipelineResponse(BaseModel):
    result: Any

class BatchPipelineResponse(BaseModel):
    results: List[Any]

@app.post("/predict", response_model=None)  # Dynamic response model
async def predict(request: PredictionRequest):
    """
    Get predictions for a single text using the specified pipeline
    """
    is_batch = isinstance(request.text, list)
    texts = request.text if is_batch else [request.text]

    if len(texts) > 32: 
        raise HTTPException(status_code=400, detail="Batch size should not exceed 32")

    model_info = get_model(request.model, request.pipeline)
    tokenizer = model_info["tokenizer"]
    model = model_info["model"]

    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)

    result=None
    
    if model_info["pipeline"] == "embedding":
        input_ids = tokenizer._tokenizer(
            texts, 
            return_tensors="mlx", 
            padding=True, 
            truncation=True, 
            max_length= max_position_embeddings
        )
        outputs = model(input_ids)
        embeddings=outputs['embeddings'] # by default, output is returned as a dict. if not, outputs[0] is the pooled_output and outputs[1]

        result = {
            "embeddings":embeddings.tolist()
        }
    
    elif model_info["pipeline"] == "sentence-transformers":
        input_ids = tokenizer._tokenizer(
            texts, 
            return_tensors="mlx", 
            padding=True, 
            truncation=True, 
            max_length= max_position_embeddings
        )
        
        reference_ids = tokenizer._tokenizer(
            request.reference_text, 
            return_tensors="mlx", 
            padding=True, 
            truncation=True, 
            max_length= max_position_embeddings
        )

        # Generate embeddings
        outputs = model(
            input_ids['input_ids'], 
            reference_ids['input_ids'],
            attention_mask=input_ids['attention_mask'],
            reference_attention_mask=reference_ids['attention_mask']
        )
    
        similarities = outputs['similarities'] # by default returned as a dictionary (use embeddings=outputs[1] otherwise)

        result =  {
            "similarities": similarities.tolist()
        }
    
    elif model_info["pipeline"] == "zero-shot-classification":
        if isinstance(request.label_candidates, dict):
            categories = "\n".join([f"{i}: {k} ({v})" for i, (k, v) in enumerate(request.label_candidates.items())])
        else:
            categories = "\n".join([f"{i}: {label}" for i, label in enumerate(request.label_candidates)])

        
        classification_inputs = []

        for text in texts:
            # Use in the f-string
            classification_input = f"""You will be given a text and categories to classify the text.

                {text}

                Read the text carefully and select the right category from the list. Only provide the index of the category:
                {categories}

                ANSWER: [unused0][MASK]
            """
            classification_inputs.append(classification_input)

        input_ids = tokenizer._tokenizer(
            classification_inputs, 
            return_tensors="mlx", 
            padding=True, 
            truncation=True, 
            max_length= max_position_embeddings
        )

        # Forward pass
        outputs = model(
            input_ids=input_ids['input_ids'],
            attention_mask=input_ids.get('attention_mask', None),
            return_dict=True
        )

        # Get the predictions for the masked token
        predictions = outputs["logits"]
        mask_token_id = tokenizer.mask_token_id
        mask_positions = mx.argmax(input_ids['input_ids'] == mask_token_id, axis=1)

        batch_results = []
        for i in range(len(classification_inputs)):
            # Find mask position for this example
            mask_position = mask_positions[i].item()
            
            # Get predictions for the masked token
            masked_token_predictions =  predictions[i, mask_position]
            
            # Process as before
            probs = mx.softmax(masked_token_predictions)
            top_k = min(5, len(request.label_candidates))
            sorted_indices = mx.argsort(probs)[::-1]
            top_indices = sorted_indices[:top_k].astype(mx.int32)
            top_probs = probs[top_indices]
            
            classification_result = [[tokenizer.decode([idx]), logit] for idx, logit in zip(top_indices.tolist(), top_probs.tolist())]
            batch_results.append(classification_result)

        result =  {
            "classification": batch_results
        }

    mx.clear_cache()
    gc.collect()  
    
    return result

@app.get("/pipelines")
async def list_pipelines():
    """
    List all available pipelines
    """
    return {
        "available_pipelines": list(PIPELINES),
        "loaded_pipeline": model_cache.get("pipeline"),
        "loaded_model" : model_cache.get("model_name")
    }

@app.get("/health")
async def health_check():
    """
    Check if the server is healthy
    """
    return {
        "status": "healthy", 
        "available_pipelines": list(PIPELINES),
        "loaded_pipeline": model_cache.get("pipeline"),
        "loaded_model" : model_cache.get("model_name")
    }

@app.post("/unload")
async def unload_model():
    """
    Unload the currently loaded model from memory
    """
    global model_cache
    
    if not model_cache:
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}
    
    # Store what was unloaded for the response
    unloaded_info = {
        "model_name": model_cache.get("model_name"),
        "pipeline": model_cache.get("pipeline")
    }
    
    # Clear the model cache
    model_cache = {}
    
    # Force garbage collection to free memory
    gc.collect()
    
    return {
        "status": "success", 
        "message": f"Model unloaded successfully",
        "unloaded": unloaded_info
    }

if __name__ == "__main__":
    uvicorn.run("utils.server:app", host="0.0.0.0", port=8000, workers=1)


### EXAMPLE
'''
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": [
      "The new MacBook Pro with M3 chip delivers exceptional performance and battery life.",
      "I was really disappointed with the customer service at that restaurant.",
      "This movie has beautiful cinematography but the plot is confusing.",
      "The aging of the population is the archetype of an unpleasant truth for mainstream media readers and for voters, which does not encourage anyone to put it on the table. Age pyramids, birth and fertility indicators, and celibacy rates in all developed countries indicate that the situation is worrying. Among these countries, some managed to stay on-track until about 10 years ago but they eventually fell into line."
    ],
    "model": "answerdotai/ModernBERT-Large-Instruct",
    "pipeline": "zero-shot-classification",
    "label_candidates": {
        "artificial intelligence": "The study of computer science that focuses on the creation of intelligent machines that work and react like humans.",
        "physics": "The study of matter, energy, and the fundamental forces of nature.",
        "society" : "The aggregate of people living together in a more or less ordered community.",
        "biology" : "The study of living organisms, divided into many specialized fields that cover their morphology, physiology, anatomy, behavior, origin, and distribution.",
        "environment" : "The surroundings or conditions in which a person, animal, or plant lives or operates.",
        "health" : "The state of being free from illness or injury.",
        "finance" : "The management of large amounts of money, especially by governments or large companies."
    }
  }'

'''