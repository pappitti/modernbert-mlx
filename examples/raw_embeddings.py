import mlx.core as mx
from utils.utils import load

'''
answerdotai/ModernBERT-base IS NOT TRAINED FOR SENTENCE SIMILARITY
MODELS TRAINED TO GENERATE EMBEDDINGS FOR SENTENCE TRANSFORMERS WILL AUTOMATICALLY
USE THE SENTENCE-TRANSFORMERS PIPELINE (SEE README)
'''

tested_models = [
    "nomic-ai/modernbert-embed-base",
    "answerdotai/ModernBERT-base"
]

def main():
    # Load the model and tokenizer
    model, tokenizer = load(
        "answerdotai/ModernBERT-base", 
        pipeline="embeddings", # models trained for sentence similarity will automatically use the "sentence-transformers" pipeline
    ) 
    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)

    def get_embedding(text, model, tokenizer):
        print(text)
        input_ids = tokenizer.encode(
            text, 
            return_tensors="mlx", 
            padding=True, 
            truncation=True, 
            max_length= max_position_embeddings
        )
        outputs = model(input_ids)
        embeddings=outputs['embeddings'] # by default, output is returned as a dict. if not, outputs[0] is the pooled_output and outputs[1]

        return embeddings

    # Sample texts
    texts = [
        "I like grapes",
        "I like fruits",
        "The slow green turtle crawls under the busy ant.",
        "Sand!",
    ]

    # Generate embeddings
    embeddings = [get_embedding(text, model, tokenizer) for text in texts]

    def cosine_similarity(a, b):
        # Compute dot product and magnitudes using MLX operations
        dot_product = mx.sum(a * b)
        norm_a = mx.sqrt(mx.sum(a * a)) 
        norm_b = mx.sqrt(mx.sum(b * b)) 
        return dot_product / (norm_a * norm_b) 

    # Calculate similarity matrix
    n = len(embeddings)
    similarity_matrix = mx.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

    # Print the similarity matrix as a table
    print("\nCosine Similarity Matrix:")
    print("-" * 40)
    print("    ", end="")
    for i in range(n):
        print(f"Text {i:<8}", end="")
    print("\n" + "-" * 40)

    for i in range(n):
        print(f"Text {i:<3}", end=" ")
        for j in range(n):
            print(f"{float(similarity_matrix[i, j]):8.4f}", end="")
        print()

if __name__ == "__main__":
    main()
