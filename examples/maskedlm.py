import mlx.core as mx
from utils.utils import load

tested_models = [
    "answerdotai/ModernBERT-base"
]

def main():
    # Load the model and tokenizer
    model, tokenizer = load(
        "answerdotai/ModernBERT-base", 
        pipeline='masked-lm' # if the config file includes the architecture "ModernBertForMaskedLM", the pipeline will be identified automatically so no need to specify it
    )

    # Prepare the input text
    text = "The capital of France is [MASK]."

    # Tokenize the input
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])  # Add batch dimension

    # Find the position of the mask token
    mask_token_id = tokenizer.mask_token_id
    mask_position = tokens.index(mask_token_id)

    # Forward pass
    outputs = model(input_ids=input_ids, return_dict=True)

    # Get the predictions for the masked token
    predictions = outputs["logits"]
    masked_token_predictions = predictions[0, mask_position]

    # Get the top 5 predictions
    probs = mx.softmax(masked_token_predictions)
    top_k = 5

    # Sort in descending order and get top k
    sorted_indices = mx.argsort(probs)[::-1]
    top_indices = sorted_indices[:top_k].astype(mx.int32)
    top_probs = probs[top_indices]

    # Print results
    print("\nTop 5 predictions for the masked token:")
    for idx, logit in zip(top_indices.tolist(), top_probs.tolist()):
        token = tokenizer.decode([idx])
        print(f"{token}: {logit:.3f}")


if __name__ == "__main__":
    main()
