import mlx.core as mx
from utils.utils import load

"""
I was a little confused about zero-shot classification based on different code bases I've looked at
Some implementations are closer to sentence similarity than it is to text classification
HF transformers has a pipeline for zero-shot classification that is closer to text classification (NLI with labels)
I went with a third route here as per this paper by answerai: https://arxiv.org/html/2502.03793v2
"""

tested_models = [
    "answerdotai/ModernBERT-Large-Instruct",
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
        "answerdotai/ModernBERT-Large-Instruct",
        pipeline='zero-shot-classification') 
    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)

    # Prepare the input text
    text_to_classify = "ModernBERT is a modernized bidirectional encoder-only Transformer model (BERT-style) pre-trained on 2 trillion tokens of English and code data with a native context length of up to 8,192 tokens"

    text_to_classify = "The aging of the population is the archetype of an unpleasant truth for mainstream media readers and for voters, which does not encourage anyone to put it on the table. Age pyramids, birth and fertility indicators, and celibacy rates in all developed countries indicate that the situation is worrying. Among these countries, some managed to stay on-track until about 10 years ago but they eventually fell into line."

    text = f"""You will be given a text and categories to classify the text.\n\nTEXT: {text_to_classify}\n\nChoose the right category for the text from the following:\n{[f"{i} : {k} ({v})\n" for i,[k,v] in enumerate(label_candidates.items())] if label_candidates is dict else [f"{i} : {label}\n" for i,label in enumerate(label_candidates)]}\n\nANSWER: [MASK]"""

    # Tokenize the input
    tokens = tokenizer.encode(
        text, 
        return_tensors="mlx", 
        padding=True, 
        truncation=True, 
        max_length= max_position_embeddings
    )

    # Find the position of the mask token
    mask_token_id = tokenizer.mask_token_id
    mask_position = mx.argmax(tokens == mask_token_id)

    # Forward pass
    outputs = model(
        input_ids=tokens,
        return_dict=True
    )

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
