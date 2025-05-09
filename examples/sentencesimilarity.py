import mlx.core as mx
from utils.utils import load

tested_models = [
    "nomic-ai/modernbert-embed-base",
    "tasksource/ModernBERT-base-embed",
    "makiart/ft-modern-bert-emb-all-nli"
]

def main():
    # Load the model and tokenizer
    model_name = "nomic-ai/modernbert-embed-base"  
    model, tokenizer = load(
        model_name, 
        pipeline="sentence-similarity" # or "sentence-transformers" if sentence-transformers model is used
    ) # if the model_path file includes "config_sentence_transformers.json", the "sentence-transformers" pipeline will be identified automatically so no need to specify it
    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)
    print(max_position_embeddings)

    texts = [
        "What is TSNE?",
        "Who is Laurens van der Maaten?",
        "I like grapes",
        "Grandma's cat got bored last winter."
    ]

    reference_texts = [
        "I like fruits",
        "The slow green turtle crawls under the busy ant.",
        "Sand!",
        "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
        "The study of computer science that focuses on the creation of intelligent machines that work and react like humans.",
        "The study of matter, energy, and the fundamental forces of nature.",
        "The aggregate of people living together in a more or less ordered community.",
    ]


    # _tokenizer is used for batch processing. 
    # Unlike the encode method, it does not return a list of tokens but a dictionary with the following keys that are used for the forward pass:
    # - input_ids: The token IDs of the input text.
    # - attention_mask: A mask to avoid performing attention on padding token indices.
    input_ids = tokenizer._tokenizer(
        texts, 
        return_tensors="mlx", 
        padding=True, 
        truncation=True, 
        max_length= max_position_embeddings
    )

    reference_ids = tokenizer._tokenizer(
        reference_texts, 
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

    print(texts)
    print("    ")
    print(reference_texts)

    # Print the similarity matrix as a table
    print(f"\nCosine Similarity Matrix: {model_name}")
    print("-" * 50)
    for i, row in enumerate(similarities):
        # Format each number to 4 decimal places
        formatted_row = [f"{x:.4f}" for x in row]
        print(f"Text {i}: {formatted_row}")

if __name__ == "__main__":
    main()
