import mlx.core as mx
from utils import load

def main():
    # Load the model and tokenizer
    model, tokenizer = load("andriadze/modernbert-chat-moderation-X-V2", pipeline='text-classification') #argilla/ModernBERT-domain-classifier
    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)

    # Prepare the input text
    text = "Grandma's cat felt very lonely during the winter holidays."

    # Tokenize the input
    tokens = tokenizer.encode(
        text, 
        return_tensors="mlx", 
        padding=True, 
        truncation=True, 
        max_length= max_position_embeddings
    )
    # input_ids = tokens[None,:] # Add batch dimension

    # Forward pass
    outputs = model(input_ids=tokens, return_dict=True)

    # Get the processed predictions for the first (and only) item in batch
    predictions = outputs["probs"][0] # Shape: (num_label,)

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
