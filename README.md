# ModernBERT-MLX (WIP)

Implementation of [ModernBERT](https://arxiv.org/abs/2412.13663) in MLX  

## Installation
```bash
git clone https://github.com/pappitti/modernbert-mlx.git
cd modernbert-mlx
uv sync
```

## Limitations of MLX
- flash attention 2 is not supported which means some features cannot be implemented at this stage
- not a limitation of MLX per se but the unpadding was not implemented here for now (linked to flash attention 2)

## Pipelines
In order to use different model classes for different use-cases, a concept of pipeline was introduced to load the model.  
```python
from utils import load
model, tokenizer = load("answerdotai/ModernBERT-base", pipeline='masked-lm')
```  
  
Pipeline list : 
- "embeddings": uses the Model class. Returns the pooled, unnormalized embeddings of the input sequence. Pooling strategy (CLS or mean) is defined by config file.  
See examples/raw_embeddings.py  
  
- "sentence-similarity" : uses the ModelForSentenceSimilarity class, which extends Model, and returns the similarity matrix, using cosine similarity, between input sequences and reference sequences 
  
- "sentence-transformers": uses the ModelForSentenceTransformers class, which extends ModelForSentenceSimilarity. The only difference is weight sanitation as sentence transformers parameters keys are specific.  
See examples/sentencetransformers.py  
  
- "zero-shot-classification" (WIP): uses the ModelForZeroShotClassification class, which extends Model, and returns probabilities of labels for the sequence. There are other interpretrations of what zero-shot classification means, notably classifications that require fixed labels. here, labels must be provided in the config file as a list or as a dictionary {label:description} as a label_candidates parameter in the config file. 
See examples/zeroshot.py 
  
- "masked-lm": uses the ModelForMaskedLM. Returns logits for all tokens in the input sequence. For now, filtering fo rthe masked token and softmax are handled outside the pipeline (see tests_maskedlm.py). It probably makes sense to integrate this in the pipeline.  
See examples/maskedlm.py  
  
- "text-classification": uses the ModelForSequenceClassification class. Returns probabilities of labels for the sequence. Classification can be a regression (untested for now), binary classification (untested for now) or multilabel classification. For multilabel the config file must contain an id2label dictionary. 
See examples/textclassification.py
  
- "token-classification" (not tested): uses the ModelForTokenClassification class. Returns probabilities of labels for each token in the sequence.

Running example file : `uv run python -m examples.raw_embeddings`

## Work in Progress
- Current implementation has yielded encouraging results for MaskedLM tasks using in [this HF checkpoint](https://huggingface.co/answerdotai/ModernBERT-base). Only at inference
- models/modernbert includes a Model class meant to return sequence embeddings. Given most checkpoints trained for sentence similarity use a config for Sentence Transformers, an additional class, ModelForSentenceTransformers, was created specifically to sanitize the weights appropriately. Tested for various models listed in examples/sentencetransformers.py  notably [nomic modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base). Only at inference
- ModelForSequenceClassification tested for multi-label classification with supportive results for various models listed in examples/textclassification.py. Single-label classification not tested yet. Only at inference  
- Zero-shot classification being closer to similarity, a dedicated pipeline uses a ModelForZeroShotClassification which inherits from Model just like ModelForSentenceTransformers. Tested with [nomic modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base). Not working for the moment    
- ModelForTokenClassification is only a placeholder at this stage.

## Next Steps
- Continue work on training
- train classifiers and sentence embeddings model to check Model, ModelForSentenceTransformers, ModelForSequenceClassification and ModelForTokenClassification
- clean the code and improve consistency across model classes for inference and training
- add other models that are relevant for these tasks (deberta?)

## Inspiration
- [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py), instrumental to this project
- [MLX Examples](https://github.com/ml-explore/mlx-examples) by Apple, is the source of the utils for this project (see licence)
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) by Prince Canuma whose project, supporting BERT and xml-roberta, has been more than helpful to get started with this one 