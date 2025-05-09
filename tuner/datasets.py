import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset as hf_load_dataset, Features, DatasetDict
import mlx.core as mx

class DatasetArgs:
    """Arguments for dataset loading"""

    def __init__(self, data: str, task_type: str, train : bool, text_field: Optional[str] = "text", label_field: Optional[str] = "label"):
        self.data = data
        self.task_type = task_type
        self.train = train
        self.text_field = text_field
        self.label_field = label_field

class Dataset:
    """Dataset class for ModernBERT that handles various tasks and data sources"""
    
    def __init__(self, data: List[Dict[str, Any]], task_type: str, label_names:List[str]=[]):
        self.data = data
        self.task_type = task_type
        self.labels = label_names
        self.label2id = None
        self.id2label = None

        #load label mapping from the label_names if provided
        if len(self.labels) > 0 and task_type == "text-classification":
            try : 
                self._load_label_mapping()
            except:
                # if loading fails for some reason, continue without label mapping
                # this will be caught in the _validate_data
                pass

        self._validate_data()

        # if labels were not provided intially but labels were added via validation, create the mapping
        if len(self.labels) > 0 and not self.label2id:
            self._load_label_mapping()
        
    def _load_label_mapping(self):
        """Loads label mapping from the label list"""
        self.id2label = { k:v for k, v in enumerate(self.labels)}
        self.label2id = { v:k for k, v in enumerate(self.labels)}
    
    def _validate_data(self):
        """Ensures data format matches the task requirements"""
        for item in self.data:

            if self.task_type == "masked-lm":
                if "text" not in item:
                    raise ValueError("MLM data must contain 'text' field")
                
            elif self.task_type == "text-classification":
                if "text" not in item or "label" not in item:
                    raise ValueError("Classification data must contain 'text' and 'label' fields")
                
                label = item.get("label",None)
                # if there is no mapping already, add the label to the list
                if not self.label2id and label not in self.labels:
                    self.labels.append(label)
                elif self.label2id and label :
                    if isinstance(label, str):
                        if label not in self.label2id:
                            raise ValueError(f"Label '{label}' not found in label mapping")
                    elif isinstance(label, (int, float)):
                        if label not in self.id2label:
                            raise ValueError(f"Label '{label}' not found in label mapping")
                    else:
                        raise ValueError(f"unexpected error with label '{label}'")

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
    
    # def shuffle(self):
    #     indices = mx.random.permutation(len(self.data))
    #     indices = indices.tolist() # Convert to list for indexing
    #     shuffled_data = [self.data[i] for i in indices]
    #     return Dataset(shuffled_data, self.task_type, self.labels)

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Loads data from a JSONL file"""
    if not file_path.exists():
        return []
        
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def create_splits(data: List[Dict[str, Any]], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> Dict[str, List[Dict[str, Any]]]:
    """
    Creates train/validation/test splits from a single dataset
    
    Args:
        data: List of data examples
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        
    Returns:
        Dictionary containing the splits
    """
    # Verify ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
        raise ValueError("Split ratios must sum to 1")
        
    # Shuffle data
    data_copy = data.copy()
    mx.random.seed(42)  # For reproducibility
    mx.random.shuffle(data_copy) ### won't work
    ### TODO something like this
    # indices = range(0, len(dataset), batch_size)
    #     if shuffle:
    #         indices = mx.random.permutation(len(data))
            
    #     for i in indices:
    #         batch = data[i:i + batch_size]
    
    # Calculate split points
    n = len(data_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        "train": data_copy[:train_end],
        "validation": data_copy[train_end:val_end],
        "test": data_copy[val_end:]
    }

def load_local_dataset(data_path: Path, task_type: str) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """Loads dataset from local jsonl files"""

    # Load label mapping from config if available
    config_path = data_path / "README.md"
    labels=[]
    if config_path.exists() :
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                # extract label mapping TODO : other tasks, maybe LLM call to read the yaml file and find the path
                features = config.get('dataset_info').get("features", [])
                for feature in features:
                    if feature.get('name') == 'label':
                        label_names  = feature.get('dtype', {}).get('class_labels', {}).get('names', [])
                        if label_names:
                            labels = [v for v in label_names.values()] 
        
        except Exception as e:
            raise ValueError(f"Error loading label mapping from config: {e}")

    # Check which splits exist
    train_path = data_path / "train.jsonl"
    valid_path = data_path / "valid.jsonl"
    test_path = data_path / "test.jsonl"
    
    has_train = train_path.exists()
    has_valid = valid_path.exists()
    has_test = test_path.exists()
    
    if not has_train and (data_path / "data.jsonl").exists():
        # Single file case
        print("Found single data.jsonl file. Creating splits...")
        data = load_jsonl(data_path / "data.jsonl")
        splits = create_splits(data)
        
    elif not has_train:
        raise ValueError("No training data found. Expect either train.jsonl or data.jsonl")
        
    else:
        # Load available splits
        splits = {
            "train": load_jsonl(train_path),
            "validation": load_jsonl(valid_path) if has_valid else None,
            "test": load_jsonl(test_path) if has_test else None
        }
        
        # Create missing splits if needed
        if not has_valid and not has_test:
            print("Creating validation and test splits from train...")
            new_splits = create_splits(splits["train"])
            splits = new_splits
            
        elif not has_valid and has_test:
            print("Creating validation split from train...")
            train_data = splits["train"]
            # Calculate validation size to maintain proportions
            val_size = 0.176  # 0.176 of remaining data ≈ 15% of total
            train_val_splits = create_splits(
                train_data,
                train_ratio=1-val_size,
                val_ratio=val_size,
                test_ratio=0
            )
            splits["train"] = train_val_splits["train"]
            splits["validation"] = train_val_splits["validation"]
            
        elif has_valid and not has_test:
            print("Creating test split from train...")
            train_data = splits["train"]
            # Calculate test size to maintain proportions
            test_size = 0.176  # 0.176 of remaining data ≈ 15% of total
            train_test_splits = create_splits(
                train_data,
                train_ratio=1-test_size,
                val_ratio=0,
                test_ratio=test_size
            )
            splits["train"] = train_test_splits["train"]
            splits["test"] = train_test_splits["test"]
    
    # Print split sizes
    total = sum(len(split) for split in splits.values() if split is not None)
    print("\nFinal split sizes:")
    for split_name, split_data in splits.items():
        if split_data is not None:
            count = len(split_data)
            percentage = count / total * 100
            print(f"{split_name}: {count} examples ({percentage:.1f}%)")
    
    # Create Dataset objects
    datasets = {}
    for split_name in ["train", "validation", "test"]:
        if splits[split_name]:
            datasets[split_name] = Dataset(
                splits[split_name],
                task_type,
                labels
            )
        else:
            datasets[split_name] = None
    
    return (
        datasets["train"],
        datasets["validation"],
        datasets["test"]
    )

def process_hf_dataset(
        dataset, 
        task_type: str, 
        text_field: str = "text", 
        label_field: str = "label"
) -> List[Dict[str, Any]]:
    """Converts HuggingFace dataset to ModernBERT format"""
    processed_data = []
    
    for item in dataset:
        if task_type == "masked-lm":
            processed_data.append({"text": item.get(text_field)})
        elif task_type == "text-classification":
            processed_data.append({
                "text": item.get(text_field),
                "label": item.get(label_field)
            })
        elif task_type == "token-classification":
            processed_data.append({
                "text": item.get(text_field),
                "labels": item.get(label_field)
            })
        elif task_type == "sentence-transformers":
            # Assuming the dataset has sentence pairs and scores
            processed_data.append({
                "text": [item.get("sentence1"), item.get("sentence2")],
                "similarity_score": item.get("score")
            })
    
    return processed_data

def load_hf_dataset(
    dataset_name: str, 
    task_type: str, 
    text_field: str = "text", 
    label_field: str = "label"
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads and processes a HuggingFace dataset with automatic splitting if needed.
    
    Args:
        dataset_name (str): Name of the dataset on HuggingFace
        task_type (str): Type of task (e.g., "text_classification")
        text_field (str): Name of the text field in the dataset
        label_field (str): Name of the label field in the dataset
        
    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets
    """
    # Load the dataset
    dataset = hf_load_dataset(dataset_name)
    
    if "train" not in dataset:
        raise ValueError("Training split not found in dataset")
    
    # Extract label information
    features = dataset["train"].features
    feature_label = features.get(label_field, {})
    label_names = feature_label.names if feature_label.names else [] 

    # Check which splits exist
    has_validation = "validation" in dataset
    has_test = "test" in dataset
    
    # Handle different split combinations
    if not has_validation and not has_test:
        print("Dataset only has train split. Creating validation and test splits...")
        
        # First split: separate test set (15% of total)
        splits = dataset["train"].train_test_split(
            test_size=0.15,
            shuffle=True,
            seed=42
        )
        train_and_val = splits["train"]
        test_split = splits["test"]
        
        # Second split: separate validation set (15% of remaining data)
        splits = train_and_val.train_test_split(
            test_size=0.176,  # 0.176 of 85% ≈ 15% of total
            shuffle=True,
            seed=42
        )
        train_split = splits["train"]
        val_split = splits["test"]
        
        dataset = {
            "train": train_split,
            "validation": val_split,
            "test": test_split
        }
        
    elif not has_validation and has_test:
        print("Dataset has train and test splits. Creating validation split from train...")
        
        # Calculate validation size to maintain proportions
        total_size = len(dataset["train"]) + len(dataset["test"])
        target_val_ratio = 0.15
        
        # Split training data to create validation
        val_size = target_val_ratio / (1 - target_val_ratio)  # Relative to remaining train
        splits = dataset["train"].train_test_split(
            test_size=val_size,
            shuffle=True,
            seed=42
        )
        
        dataset = {
            "train": splits["train"],
            "validation": splits["test"],
            "test": dataset["test"]
        }
        
    elif has_validation and not has_test:
        print("Dataset has train and validation splits. Creating test split from train...")
        
        # Calculate test size to maintain proportions
        target_test_ratio = 0.15
        
        # Split training data to create test
        test_size = target_test_ratio / (1 - target_test_ratio)  # Relative to remaining train
        splits = dataset["train"].train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=42
        )
        
        dataset = {
            "train": splits["train"],
            "validation": dataset["validation"],
            "test": splits["test"]
        }
    
    # Print final split sizes
    total = sum(len(dataset[split]) for split in dataset.keys())
    print("\nFinal split sizes:")
    for split in dataset.keys():
        count = len(dataset[split])
        percentage = count / total * 100
        print(f"{split}: {count} examples ({percentage:.1f}%)")
    
    # Process each split
    processed_splits = {}
    for split in ["train", "validation", "test"]:
        if split in dataset:
            data = process_hf_dataset(
                dataset[split],
                task_type,
                text_field,
                label_field
            )
            processed_splits[split] = Dataset(data, task_type, label_names)
        else:
            processed_splits[split] = None
    
    return (
        processed_splits["train"],
        processed_splits["validation"],
        processed_splits["test"]
    )

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