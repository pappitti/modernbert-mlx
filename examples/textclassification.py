import mlx.core as mx
from utils.utils import load

tested_models = {
    "multilabel":[
        "clapAI/modernBERT-base-multilingual-sentiment",
        "argilla/ModernBERT-domain-classifier",
        "andriadze/modernbert-chat-moderation-X-V2",
    ],
    "regression":[
        "Forecast-ing/modernBERT-content-regression" # Used it to confirm that regression works but I don't recommend this specific checkpoint
    ]   
}

def main():
    
    is_regression = False # Set to True for regression models
    
    # Load the model and tokenizer
    model, tokenizer = load(
        "ModernBERT-base_text-classification/checkpoint-66",
        model_config={"is_regression":is_regression}, 
        pipeline='text-classification'
    ) 
    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)

    # Prepare the input text
    text = "The study of computer science that focuses on the creation of intelligent machines that work and react like humans."

    # Tokenize the input
    tokens = tokenizer.encode(
        text, 
        return_tensors="mlx", 
        padding=True, 
        truncation=True, 
        max_length= max_position_embeddings
    )

    # Forward pass
    outputs = model(input_ids=tokens, return_dict=True)

    # Get the processed predictions for the first (and only) item in batch
    predictions = outputs["probabilities"][0] # Shape: (num_label,)

    top_k = 5

    # Sort in descending order and get top k
    sorted_indices = mx.argsort(predictions)[::-1]
    top_indices = sorted_indices[:top_k]
    top_probs = predictions[top_indices]

    id2label = model.config.id2label

    print(text)

    # Print results
    print("\nTop 5 predictions for classification:")
    for idx, logit in zip(top_indices.tolist(), top_probs.tolist()):
        token = id2label[str(idx)]
        print(f"{token}: {logit:.3f}")

if __name__ == "__main__":
    main()