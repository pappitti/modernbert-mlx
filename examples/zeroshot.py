import mlx.core as mx
from utils.utils import load

"""
I am a little confused about zero-shot classification based on different code bases I've looked at
This implementation is closer to sentence similarity than it is to text classification
HF transformers has a pipeline for zero-shot classification that is closer to text classification (NLI with labels)
I keep this in for now but I might remove it later
"""

tested_models = [
    "nomic-ai/modernbert-embed-base",
]

def main():
    # define the label candidates
    label_candidates = {
        "artificial intelligence": "The study of computer science that focuses on the creation of intelligent machines that work and react like humans.",
        "physics": "The study of matter, energy, and the fundamental forces of nature.",
        "society" : "The aggregate of people living together in a more or less ordered community.",
        "biology" : "The study of living organisms, divided into many specialized fields that cover their morphology, physiology, anatomy, behavior, origin, and distribution.",
        "environment" : "The surroundings or conditions in which a person, animal, or plant lives or operates.",
        "health" : "The state of being free from illness or injury.",
        "finance" : "The management of large amounts of money, especially by governments or large companies.",
    }

    # Load the model and tokenizer
    model, tokenizer = load(
        "nomic-ai/modernbert-embed-base",
        pipeline='zero-shot-classification') 
    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)

    # Prepare the input text
    text = "ModernBERT is a modernized bidirectional encoder-only Transformer model (BERT-style) pre-trained on 2 trillion tokens of English and code data with a native context length of up to 8,192 tokens"

    # Tokenize the input
    tokens = tokenizer.encode(
        text, 
        return_tensors="mlx", 
        padding=True, 
        truncation=True, 
        max_length= max_position_embeddings
    )

    if type(label_candidates) is dict:
        label_defs = list(label_candidates.values())
        label_keys = list(label_candidates.keys())
    else:
        label_defs = label_keys = label_candidates
    encoded_labels = tokenizer._tokenizer(
        label_defs, 
        return_tensors="mlx", 
        padding=True, 
        truncation=True, 
        max_length= max_position_embeddings
    )

    # Forward pass
    outputs = model(
        input_ids=tokens,
        label_candidates=encoded_labels['input_ids'],
        label_candidates_attention_mask=encoded_labels['attention_mask'],
        multi_label=False, # returns probabilities if true, similarities if false
        return_dict=True
    )

    # Get the processed predictions 
    predictions = outputs["output"][0] # Shape: (batch_size, num_label,)

    top_k = 5

    # Sort in descending order and get top k
    sorted_indices = mx.argsort(predictions)[::-1]
    top_indices = sorted_indices[:top_k]
    top_probs = predictions[top_indices]

    print(text)

    # Print results
    print("\nTop 5 predictions for classification:")
    for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
        label = label_keys[idx]
        print(f"{label}: {prob:.3f}")

if __name__ == "__main__":
    main()
