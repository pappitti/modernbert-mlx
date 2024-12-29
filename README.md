# ModernBERT-MLX (WIP)

Implementation of [ModernBERT](https://arxiv.org/abs/2412.13663) in MLX  

## Limitations of MLX
- flash attention 2 is not supported which means some features cannot be implemented at this stage
- not a limitation of MLX per se but the unpadding was not implemented here for now (linked to flash attention 2)

## Progress
- Current implementation has yielded encouraging results for MaskedLM tasks using in [this HF checkpoint](https://huggingface.co/answerdotai/ModernBERT-base) and ModelForMaskedLM in models/modernbert. Only at inference
- models/modernbert includes a Model class (WIP) meant to return sequence embeddings. Given checkpoint is not trained for sentence embeddings, testing is challenging. 
- ModelForSequenceClassification and ModelForTokenClassification not tested. Classes inspired by [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py)

## Next Steps
- Start work on training
- train classifiers and sentence embeddings model to check Model, ModelForSequenceClassification and ModelForTokenClassification
- add other models that are relevant for these tasks (deberta?)

## Inspiration
- [MLX Examples](https://github.com/ml-explore/mlx-examples) by Apple, is the source of the utils for this project (see licence)
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) by Prince Canuma whose project, supporting BERT and xml-ropert, has been more than helpful to get started with this one 