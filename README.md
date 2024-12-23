# ModernBERT-MLX (WIP)

Implementation of [ModernBERT](https://arxiv.org/abs/2412.13663) in MLX  

## Limitations of MLX
- flash attention 2 is not supported which means some features cannot be implemented at this stage

## Progress
- Current implementation has yielded encouraging results for MaskedLM tasks using in [this HF checkpoint](https://huggingface.co/answerdotai/ModernBERT-base) and ModelForMaskedLM in models/modernbert. Only at inference
- models/modernbert includes a Model class (WIP) meant to return sequence embeddings. Initial tests are not conclusive.

## Next Steps
- finalizing the Model class
- Start work on training
- add new model classes such as ModernBertForSequenceClassification or ModernBertForTokenClassification inspired by [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py)
- adding other models that are relevant for these tasks

## Inspiration
- [MLX Examples](https://github.com/ml-explore/mlx-examples) by Apple, is the source of the utils for this project (see licence)
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) by Prince Canuma whose project, supporting BERT and xml-roperta has been more than helpful to get started witht this one 