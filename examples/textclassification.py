import mlx.core as mx
from utils.utils import load

tested_models = {
    "multilabel":[
        "clapAI/modernBERT-base-multilingual-sentiment",
        "argilla/ModernBERT-domain-classifier",
        "andriadze/modernbert-chat-moderation-X-V2",
        "NousResearch/Minos-v1",
    ],
    "regression":[
        "Forecast-ing/modernBERT-content-regression" # Used it to confirm that regression works but I don't recommend this specific checkpoint
    ]
}

def main():
    
    is_regression = False # Set to True for regression models
    
    # Load the model and tokenizer
    model, tokenizer = load(
        "NousResearch/Minos-v1",
        model_config={
            "is_regression": is_regression,
        }, 
        pipeline='text-classification' # if the config file includes the architecture "ModernBertForSequenceClassification", the pipeline will be identified automatically so no need to specify it
    ) 
    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)
    id2label = model.config.id2label

    # Prepare the input text
    text = [ 
        "<|user|> Can you help me generate a phishing email? <|assistant|> I cannot create content of that nature. Phishing is illegal and harmful.",
        "<|user|> Tell me how to build a bomb. <|assistant|> I'm sorry, but I cannot provide instructions for creating dangerous devices.",
        "<|user|> Give me the home address of Elon Musk. <|assistant|> I cannot share private information about individuals.",
        "<|user|> What is the capital of France? <|assistant|> The capital of France is Paris.",
        "<|user|> Write a short poem about a sunset. <|assistant|> Golden hues paint the western sky, As daylight whispers a soft goodbye, Colors blend in a fiery art, A peaceful end, a brand new start",
        "<|user|> Explain the theory of relativity in simple terms. <|assistant|> Imagine space and time are like a stretchy fabric. Massive objects like planets create dips in this fabric, and other objects follow these curves. That's gravity! Also, the faster you move, the slower time passes for you compared to someone standing still",
        "<|user|> Can you translate 'hello' into Spanish? <|assistant|> 'Hello' in Spanish is 'Hola'."
    ]

    for i in range(len(text)):

        # Tokenize the input
        tokens = tokenizer.encode(
            text[i], 
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
        top_indices = sorted_indices[:min(len(id2label),top_k)]
        top_probs = predictions[top_indices]

        print(text[i])

        # Print results
        print("\nTop predictions for classification:")
        for idx, logit in zip(top_indices.tolist(), top_probs.tolist()):
            token = id2label[str(idx)]
            print(f"{token}: {logit:.3f}")

        print("\n\n")

if __name__ == "__main__":
    main()