# ModernBERT-MLX (WIP)

Implementation of [ModernBERT](https://arxiv.org/abs/2412.13663) in MLX  

## Limitations of MLX
- flash attention 2 is not supported which means some features cannot be implemented at this stage
- not a limitation of MLX per se but the unpadding was not implemented here for now (linked to flash attention 2)

## Pipelines
In order to use different model classes for different use-cases, a concept of pipeline was introduced to load the model.  
```python
from utils import load
model, tokenizer = load("answerdotai/ModernBERT-base", pipeline='masked_lm')
```  
PIPELINES = [ "embeddings", "masked_lm", "text_classification", "token_classification", "sentence_transformers" ]
use respectively Model, ModelForMaskedLM, ModelForSequenceClassification (WIP), ModelForTokenClassification (WIP) and ModelForSentenceTransformers  

Examples of inference pipelines are presented in tests_maskedlm.py and tests_sentence_transformers.py

## Work in Progress
- Current implementation has yielded encouraging results for MaskedLM tasks using in [this HF checkpoint](https://huggingface.co/answerdotai/ModernBERT-base) and ModelForMaskedLM in models/modernbert. Only at inference
- models/modernbert includes a Model class meant to return sequence embeddings. Given most checkpoint trained for sentence similarity use a config for Sentence Transformers, an additional class, ModelForSentenceTransformers, was created specifically to sanitize the weights appropriately. Encouraging results for [this checkpoint](https://huggingface.co/makiart/ft-modern-bert-emb-all-nli) and [this checkpoint](https://huggingface.co/tasksource/ModernBERT-base-embed), only at inference
- ModelForSequenceClassification and ModelForTokenClassification not tested. Classes inspired by [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py) 

## Next Steps
- Start work on training
- train classifiers and sentence embeddings model to check Model, ModelForSentenceTransformers, ModelForSequenceClassification and ModelForTokenClassification
- add other models that are relevant for these tasks (deberta?)

## Inspiration
- [MLX Examples](https://github.com/ml-explore/mlx-examples) by Apple, is the source of the utils for this project (see licence)
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) by Prince Canuma whose project, supporting BERT and xml-roberta, has been more than helpful to get started with this one 