import mlx.core as mx
from utils import load

def main():
    # Load the model and tokenizer
    model_name = "nomic-ai/modernbert-embed-base" #makiart/ft-modern-bert-emb-all-nli, tasksource/ModernBERT-base-embed, nomic-ai/modernbert-embed-base 
    model, tokenizer = load(model_name, pipeline="sentence-transformers") 
    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)
    print(max_position_embeddings)

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
        embeddings=outputs["embeddings"] #will be returned as a dictionary by default (use embeddings=outputs[0] otherwise)

        return embeddings

    # Sample texts
    # texts = [
    #     "I like grapes",
    #     "I like fruits",
    #     "The slow green turtle crawls under the busy ant.",
    #     "Sand!",
    # ]

    texts = [
        "search_query: What is TSNE?",
        "search_query: Who is Laurens van der Maaten?",
        "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
    ]

    # Generate embeddings
    embeddings = [get_embedding(text, model, tokenizer) for text in texts]

    def cosine_similarity(a, b):
        # Compute dot product and magnitudes using MLX operations
        dot_product = mx.sum(a * b)
        # norm_a = mx.sqrt(mx.sum(a * a)) ## removed because already normalized in Model
        # norm_b = mx.sqrt(mx.sum(b * b)) ## removed because already normalized in Model
        return dot_product ## / (norm_a * norm_b) removed because already normalized in Model

    # Calculate similarity matrix
    n = len(embeddings)
    similarity_matrix = mx.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

    # Print the similarity matrix as a table
    print(f"\nCosine Similarity Matrix: {model_name}")
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
