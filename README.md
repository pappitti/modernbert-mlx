# ModernBERT-MLX

Implementation of [ModernBERT](https://arxiv.org/abs/2412.13663) in MLX  

🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨  
ModernBERT-MLX is no longer maintained and has been superseded by [mlx-raclate](https://github.com/pappitti/mlx-raclate), which (**R**etrieval **A**nd **C**lassification including **LATE** interaction). 
`mlx-raclate` is a versatile library built on Apple's [MLX](https://github.com/ml-explore/mlx) framework. It provides a unified interface to **train** and **run** classifiers and embedding models - including ModernBERT and Late Interaction (ColBERT-style) models - natively on macOS.  
🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨

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

If no pipeline is provided, information from the model repo will be used to infer a pipeline:  
 - if config_sentence_transformers.json exists, pipeline will automatically be set to "sentence-transformers"
 - if model architecture in config file is ModernBertForSequenceClassification, pipeline will automatically be set to "text-classification" 
 - if model architecture in config file is ModernBertForMaskedLM, pipeline will automatically be set to "masked-lm"  
If there is no match, the pipeline defaults to masked-lm (original modernBERT model).  

### Batch processing 
For batch processing, use the _tokenizer method of the tokenizer. 

Unlike the encode method, it does not return a list of tokens but a dictionary with the following keys that are used for the forward pass:  
- input_ids: The token IDs of the input text.
- attention_mask: A mask to avoid performing attention on padding token indices.

```python
input_ids = tokenizer._tokenizer(
    texts, 
    return_tensors="mlx", 
    padding=True, 
    truncation=True, 
    max_length= max_position_embeddings
)

# Generate embeddings
outputs = model(
    input_ids['input_ids'], 
    attention_mask=input_ids['attention_mask'],
)

```
  
### Pipeline list 
- "embeddings"  
Uses the Model class. Returns the pooled, unnormalized embeddings of the input sequence. Pooling strategy (CLS or mean) is defined by config file. Note that sentence transformers models will automatically switch to the "sentence-transformers" pipeline.  
See examples/raw_embeddings.py  
  
- "sentence-similarity" and "sentence-transformers"  
"sentence-similarity" uses the ModelForSentenceSimilarity class, which extends Model, and returns a dictionary with the embeddings and similarity matrix, using cosine similarity, between input sequences and reference sequences.  
"sentence-transformers" uses the ModelForSentenceTransformers class, which extends ModelForSentenceSimilarity. The only difference is weight sanitization as sentence transformers parameters keys are specific.  
See examples/sentencesimilarity.py  
  
- "masked-lm"  
Uses the ModelForMaskedLM class. Returns logits for all tokens in the input sequence. For now, filtering for the masked token and softmax are handled outside the pipeline.  
See examples/maskedlm.py  
  
- "zero-shot-classification"  
Uses the ModelForMaskedLM class. See above. The reason for using this pipeline is a recent paper by Answer.ai showing that it was a great alternative to other approaches.  
See examples/zeroshot.py  

(a ModelForZeroShotClassification class had also been prepared, which extends Model, and returns probabilities of labels for the sequence. There are other interpretrations of what zero-shot classification means, notably classifications that require fixed labels. Not sure which approach is correct. More work needed so the class is kept for future work even if it is not used. Here, labels must be provided in the config file as a list or as a dictionary {label:description} as a label_candidates parameter in the config file.)  
  
- "text-classification"  
Uses the ModelForSequenceClassification class. Returns probabilities of labels for the sequence. Classification can be a regression (untested for now), binary classification (untested for now) or multilabel classification. For multilabel the config file must contain an id2label dictionary.  
See examples/textclassification.py  
  
- "token-classification" (not tested)  
Uses the ModelForTokenClassification class. Returns probabilities of labels for each token in the sequence.  

Running example file : `uv run python -m examples.raw_embeddings` or if the venv is activated `python -m examples.raw_embeddings`

## Work in Progress 
### Models
- Current implementation has yielded encouraging results for masked LM tasks using in [this HF checkpoint](https://huggingface.co/answerdotai/ModernBERT-base). Only at inference
- Embeddings and sentence similarity (incl. sentence transformers variants) tested for various models listed in examples/sentencesimilarity.py notably [nomic modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base). Only at inference
- text-classification tested for multi-label classification and regression with supportive results for various models listed in examples/textclassification.py. Single-label classification not tested yet. Multi-label classification training works (albeit not optimized), otherwise inference only.  
- Zero-shot classification tested with [answerdotai/ModernBERT-Large-Instruct](https://huggingface.co/answerdotai/ModernBERT-Large-Instruct). WIP    
- ModelForTokenClassification is only a placeholder at this stage.

### Utils
- The dataset handling leverages HuggingFace's [Datasets](https://huggingface.co/docs/datasets/index) but requires a lot more work to address all important usecases
- Server uses a FastAPI server can be used. It has only been tested with zero-shot classification for now (both single prediction and batch prediction). Use with `uv run python -m utils.server`

## Next Steps
- Continue work on training (PoC for sequence classification, still very preliminary)
- train classifiers and sentence embeddings model to check Model, ModelForSentenceTransformers and ModelForTokenClassification
- write docs
- add other models that are relevant for these tasks (bert, xml-roberta, qwen3-embed...)

## Inspiration
- [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py), instrumental to this project
- [MLX Examples](https://github.com/ml-explore/mlx-examples) by Apple, is the source of the utils for this project (see licence)
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) by Prince Canuma whose project, supporting BERT and xml-roberta, has been more than helpful to get started with this one. I've worked on BERT and xml-roberta in a [fork of this project](https://github.com/pappitti/mlx-embeddings). Add the Qwen3-embed architecture there too.     