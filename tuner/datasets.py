import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset as hf_load_dataset
import mlx.core as mx

class Dataset:
    """Dataset class for ModernBERT that handles various tasks and data sources"""
    
    def __init__(self, data: List[Dict[str, Any]], task_type: str):
        self.data = data
        self.task_type = task_type
        self._validate_data()
        self.labels=[] ### can I use a more direct way to get the labels?
    
    def _validate_data(self):
        """Ensures data format matches the task requirements"""
        for item in self.data:

            if self.task_type == "masked-lm":
                if "text" not in item:
                    raise ValueError("MLM data must contain 'text' field")
                
            elif self.task_type == "text-classification":
                if "text" not in item or "label" not in item:
                    raise ValueError("Classification data must contain 'text' and 'label' fields")
                
                # add to labels
                if item["label"] not in self.labels:
                    self.labels.append(item["label"])

            elif self.task_type == "sentence-transformers":
                if "text" not in item or "similarity_score" not in item:
                    raise ValueError("Sentence transformer data must contain 'text' and 'similarity_score' fields")
            elif self.task_type == "token-classification":
                if "text" not in item or "labels" not in item:
                    raise ValueError("Token classification data must contain 'text' and 'labels' fields")            

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def load_local_dataset(data_path: Path, task_type: str) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """Loads dataset from local jsonl files"""
    def load_split(path):
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = [json.loads(line) for line in f]
        return Dataset(data, task_type)
    
    return (
        load_split(data_path / "train.jsonl"),
        load_split(data_path / "valid.jsonl"),
        load_split(data_path / "test.jsonl")
    )

def process_hf_dataset(dataset, task_type: str, text_field: str = "text", label_field: str = "label") -> List[Dict[str, Any]]:
    """Converts HuggingFace dataset to ModernBERT format"""
    processed_data = []
    
    for item in dataset:
        if task_type == "masked-lm":
            processed_data.append({"text": item[text_field]})
        elif task_type == "text-classification":
            processed_data.append({
                "text": item[text_field],
                "label": item[label_field]
            })
        elif task_type == "token-classification":
            processed_data.append({
                "text": item[text_field],
                "labels": item[label_field]
            })
        elif task_type == "sentence-transformers":
            # Assuming the dataset has sentence pairs and scores
            processed_data.append({
                "text": [item["sentence1"], item["sentence2"]],
                "similarity_score": item["score"]
            })
    
    return processed_data

def load_hf_dataset(dataset_name: str, task_type: str, text_field: str = "text", label_field: str = "label") -> Tuple[Dataset, Dataset, Dataset]:
    """Loads and processes a HuggingFace dataset"""
    dataset = hf_load_dataset(dataset_name)
    
    splits = {}
    for split in ["train", "validation", "test"]:
        if split in dataset:
            data = process_hf_dataset(
                dataset[split], 
                task_type,
                text_field,
                label_field
            )
            splits[split] = Dataset(data, task_type)
        else:
            splits[split] = None
            
    return splits.get("train"), splits.get("validation"), splits.get("test")

def load_dataset(args) -> Tuple[Dataset, Dataset, Dataset]:
    """Main dataset loading function that handles both local and HF datasets"""
    if not hasattr(args, "task_type"):
        raise ValueError("Must specify task_type in args")
        
    # TODO change for PIPELINE once all the pipelines trainings are implemented
    supported_tasks = ["text-classification"]
    if args.task_type not in supported_tasks:
        raise ValueError(f"Unsupported task type: {args.task_type}. Must be one of {supported_tasks}")
    
    # Handle local dataset
    if Path(args.data).exists():
        train, valid, test = load_local_dataset(Path(args.data), args.task_type)
    # Handle HuggingFace dataset
    else:
        train, valid, test = load_hf_dataset(
            args.data,
            args.task_type,
            getattr(args, "text_field", "text"),
            getattr(args, "label_field", "label")
        )
    
    # Validate required splits are present
    if args.train and train is None:
        raise ValueError("Training set required for training")
    if args.train and valid is None:
        raise ValueError("Validation set required for training")
        
    return train or [], valid or [], test or []