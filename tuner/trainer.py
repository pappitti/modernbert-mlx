import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from textwrap import dedent

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers

from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten

from .datasets import Dataset


def grad_checkpoint(model):
    """
    Apply gradient checkpointing to the model's forward pass to reduce memory usage.
    Uses MLX's checkpoint mechanism to save memory during backpropagation.
    """
    original_call = model.__call__

    def checkpointed_call(self, **kwargs):
        # Let MLX handle the parameter management, just checkpoint the function call
        return mx.checkpoint(original_call)(self, **kwargs)

    model.__call__ = checkpointed_call

def create_mlm_masks(
    input_ids: mx.array,
    tokenizer,
    mlm_probability: float = 0.15
) -> Tuple[mx.array, mx.array]:
    """
    Creates masked input and corresponding labels for MLM training.
    
    Args:
        input_ids: Original input token IDs
        tokenizer: Tokenizer with special token information
        mlm_probability: Probability of masking a token (default 15%)
    
    Returns:
        Tuple of (masked_inputs, mlm_labels) where mlm_labels contains
        the original tokens at masked positions and -100 elsewhere
    """
    # Create probability mask
    probability_matrix = mx.random.uniform(input_ids.shape) < mlm_probability
    
    # Don't mask special tokens like [CLS], [SEP], etc.
    special_tokens_mask = mx.array([
        [1 if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
         else 0 for token_id in input_sequence]
        for input_sequence in input_ids
    ])
    probability_matrix = mx.where(special_tokens_mask, 0, probability_matrix)
    
    # Create labels: -100 for unmasked tokens (ignored in loss computation)
    labels = mx.where(probability_matrix, input_ids, -100)
    
    # Decide replacement strategy for each masked token:
    # 80% [MASK], 10% random, 10% unchanged
    random_matrix = mx.random.uniform(input_ids.shape)
    
    # Indices for [MASK] tokens (80% of masked tokens)
    mask_indices = (probability_matrix) & (random_matrix < 0.8)
    # Indices for random tokens (10% of masked tokens)
    random_indices = (probability_matrix) & (random_matrix >= 0.8) & (random_matrix < 0.9)
    
    # Create masked input
    masked_inputs = input_ids.copy()
    # Replace with [MASK] token
    masked_inputs = mx.where(mask_indices, tokenizer.mask_token_id, masked_inputs)
    # Replace with random tokens
    random_tokens = mx.random.randint(
        0, tokenizer.vocab_size, 
        shape=input_ids.shape
    )
    masked_inputs = mx.where(random_indices, random_tokens, masked_inputs)
    
    return masked_inputs, labels

@dataclass
class TrainingArgs:

    def __init__(
        self,
        batch_size: int = 32,
        eval_batch_size: int = 16,
        max_length: int = 512,
        num_train_epochs: int = 3,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        output_dir: str = "outputs",
        save_total_limit: Optional[int] = None,
        grad_checkpoint: bool = True,
        push_to_hub: bool = False,
    ):
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_length = max_length
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio ### not used here but kept for later (see scheduler in utils)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.grad_checkpoint = grad_checkpoint ### mat not be necessary but helps anticipating hardware constraints
        self.push_to_hub = push_to_hub ### not used here but kept for later (see push_to_hub in utils)

class WeightChangeTracker:
    """Tracks changes in model weights over time."""
    
    def __init__(self, model):
        self.model = model
        self.initial_weights = None
        self.weight_history = {}
        self.save_initial_weights()
    
    def save_initial_weights(self):
        """Saves a copy of initial model weights."""
        self.initial_weights = {
            name: param[:]
            for name, param in tree_flatten(self.model.parameters())
        }
        
        # Initialize history
        for name in self.initial_weights:
            self.weight_history[name] = []
    
    def track_changes(self, step):
        """Records weight changes for this step."""
        current_weights = dict(tree_flatten(self.model.parameters()))
        
        for name, initial_weight in self.initial_weights.items():
            current_weight = current_weights[name]
            
            # Compute relative change
            weight_diff = mx.abs(current_weight - initial_weight)
            relative_change = mx.mean(weight_diff / (mx.abs(initial_weight) + 1e-7))
            
            self.weight_history[name].append({
                'step': step,
                'relative_change': relative_change.item(),
                'max_change': mx.max(weight_diff).item(),
                'mean_value': mx.mean(current_weight).item(),
                'std_value': mx.std(current_weight).item()
            })
    
    def report(self, last_n_steps=5):
        """Generates a report of weight changes."""
        print("\nWeight Change Report:")
        print("-" * 50)
        
        for name, history in self.weight_history.items():
            recent_changes = history[-last_n_steps:]
            if recent_changes:
                avg_change = sum(h['relative_change'] for h in recent_changes) / len(recent_changes)
                print(f"\nLayer: {name}")
                print(f"Average relative change (last {last_n_steps} steps): {avg_change:.2e}")
                print(f"Current mean value: {recent_changes[-1]['mean_value']:.2e}")
                print(f"Current std value: {recent_changes[-1]['std_value']:.2e}")

class Trainer:
    """
    A trainer for ModernBERT that adapts to the model's training objective.
    The training logic is determined by the model's class implementation.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        task_type: str,
        training_args: TrainingArgs,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        optimizer = None
    ):
        self.model = model
        self.tokenizer = tokenizer._tokenizer ### tokenizer is a wrapper around the HF tokenizer
        self.task_type = task_type
        self.args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Initialize optimizer
        self.optimizer = optimizer or mlx.optimizers.AdamW(
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay
        )
        
        # Setup training state and output directory
        self.global_step = 0
        self.epoch = 0
        self.output_dir = Path(training_args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enable gradient checkpointing if requested
        if training_args.grad_checkpoint:
            grad_checkpoint(model)
        
        # Log model type and config
        self.push_to_hub = training_args.push_to_hub
        print(f"Training {model.__class__.__name__}")
        self._save_config()

    def prepare_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        """
        Prepare inputs based on model type. The model's forward pass will handle
        the specific training objective.
        """
        # Extract texts and labels from the batch of dictionaries
        texts = [example["text"] for example in batch]
            
        # Use tokenizer with padding and truncation
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="mlx"
        )
        
        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "position_ids": mx.arange(encoded["input_ids"].shape[1])[None, :]
        }
        
        # Add task-specific labels
        ### TODO : do better than this (once all tasks are implemented)
        if self.task_type == "text-classification" :
            labels = [example["label"] for example in batch]  # These are already IDs
            if self.model.is_regression:
                model_inputs["labels"] = mx.array(labels, dtype=mx.float32)
            else:
                model_inputs["labels"] = mx.array(labels)

        elif self.task_type == "token-classification":
            labels = [example["label"] for example in batch]
            model_inputs["labels"] = mx.array(labels)

        elif self.task_type == "sentence-transformers" or self.task_type == "sentence-similarity":
            similarity_scores = [example["similarity_score"] for example in batch]
            model_inputs["similarity_score"] = mx.array(similarity_scores, dtype=mx.float32)
            ### what's the format of the reference_texts?
            ### placeholder below
            reference_texts = [example["reference_text"] for example in batch]
            reference_encoded = self.tokenizer(
                reference_texts,
                padding=True,
                truncation=True,
                max_length=self.args.max_length,
                return_tensors="mlx"
            )
            model_inputs["reference_input_ids"] = reference_encoded["input_ids"]
            model_inputs["reference_attention_mask"] = reference_encoded["attention_mask"]

        elif self.task_type == "zero-shot-classification":
            labels = [example["label"] for example in batch]
            model_inputs["labels"] = mx.array(labels)
            
        elif self.task_type == "masked-lm":
            # For MLM, we need to mask some tokens
            inputs, labels = create_mlm_masks(
                model_inputs["input_ids"],
                self.tokenizer
            )
            model_inputs["input_ids"] = inputs
            model_inputs["labels"] = labels
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
            
        # print (f"batch inputs : inputs {model_inputs["input_ids"].shape}, attention_mask {model_inputs["attention_mask"].shape}, labels {model_inputs["labels"].shape}")
        return model_inputs

    def train(self):
        """Main training loop."""
        print("Starting training...")
        trainable_params = sum(p[1].size for p in tree_flatten(self.model.trainable_parameters()))
        print(f"Trainable parameters: {trainable_params}")

        # # Initialize weight tracker
        # tracker = WeightChangeTracker(self.model)
        
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.args.num_train_epochs}")
            self._train_epoch()

            # # Track weight changes
            # tracker.track_changes(step=self.global_step)
            # tracker.report()
            
            if self.eval_dataset is not None:
                metrics = self.evaluate()
                self._save_checkpoint(metrics)
    
    def compute_loss(self, params, batch):
            # Update model parameters temporarily for forward pass
            self.model.update(params)
            outputs = self.model(**batch)
            return mx.mean(outputs["loss"])

    def _train_epoch(self):
        """Training logic for one epoch."""
        self.model.train()
        accumulated_loss = 0
        num_steps = 0
        start_time = time.time()
        
        # Create batches from dataset
        train_dataloader = self._create_dataloader(self.train_dataset, shuffle=True)
            
        for step, batch in enumerate(train_dataloader):
            # Prepare batch - model's forward pass handles the specific objective
            inputs = self.prepare_batch(batch)
            
            # Create loss function closure for this batch
            loss_fn = lambda p: self.compute_loss(p, inputs)
            
            # Compute loss and gradients
            loss, gradients = mx.value_and_grad(loss_fn)(self.model.parameters())
            
            # Update on gradient accumulation steps
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                gradients = average_gradients(gradients)
                self.optimizer.update(self.model, gradients)
                self.global_step += 1
            
            # Logging
            accumulated_loss += loss.item()
            num_steps += 1
            
            if self.global_step % self.args.logging_steps == 0:
                avg_loss = accumulated_loss / num_steps
                elapsed = time.time() - start_time
                print(f"Step {self.global_step}: "
                      f"loss = {avg_loss:.4f}, "
                      f"steps/sec = {num_steps/elapsed:.2f}")
                accumulated_loss = 0
                num_steps = 0
                start_time = time.time()
    
    def evaluate(self):
        """Evaluation loop."""
        self.model.eval()
        all_losses = []
        
        eval_dataloader = self._create_dataloader(self.eval_dataset, shuffle=False)
        
        for batch in eval_dataloader:
            inputs = self.prepare_batch(batch)
            loss = self.model(**inputs)["loss"]
            batch_loss = mx.mean(loss)
            all_losses.append(batch_loss.item())
        
        avg_loss = sum(all_losses) / len(all_losses)
        metrics = {"eval_loss": avg_loss}
        
        print(f"\nEvaluation metrics: {metrics}")
        return metrics
    
    def test(self, test_dataset=None):
        """
        Evaluate the model on the test set after training is complete.
        
        Args:
            test_dataset: Optional test dataset. If None, uses self.eval_dataset
        """
        print("\nPerforming final evaluation on test set...")
        
        # Save the model's training state
        training = self.model.training
        self.model.eval()
        
        # Use provided test dataset or fall back to eval dataset
        dataset_to_test = test_dataset or self.eval_dataset
        if dataset_to_test is None:
            raise ValueError("No test dataset provided")
        
        # Perform evaluation
        all_losses = []
        test_dataloader = self._create_dataloader(dataset_to_test, shuffle=False)
        
        for batch in test_dataloader:
            inputs = self.prepare_batch(batch)
            loss = self.model(**inputs)["loss"]
            batch_loss = mx.mean(loss)
            all_losses.append(batch_loss.item())
        
        # Compute metrics
        test_loss = sum(all_losses) / len(all_losses)
        metrics = {"test_loss": test_loss}
        
        # Save test results
        results_path = self.output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Test results: {metrics}")
        
        # Restore model's training state
        self.model.train(training)
        
        return metrics
    
    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False):
        """Create a dataloader from a HuggingFace dataset."""
        if shuffle:
            dataset = dataset.shuffle()
        
        for i in range(0, len(dataset), self.args.batch_size):
            yield dataset[i:i + self.args.batch_size]
    
    def _save_checkpoint(self, metrics: Dict[str, float]):
        """Save a model checkpoint."""
        save_path = self.output_dir / f"checkpoint-{self.global_step}"
        save_path.mkdir(exist_ok=True)

        # Save model config - using the actual model's configuration
        with open(save_path / "config.json", "w") as f:
            json.dump(self.model.config.__dict__, f, indent=2)

        # Save tokenizer json
        self.tokenizer.save_pretrained(save_path)
        
        # Save model weights
        ### sharding not necessary for small models
        weights = dict(tree_flatten(self.model.trainable_parameters())) ### need more than just the model. (everthing included?)
        mx.save_safetensors(str(save_path / "model.safetensors"), weights, metadata={"format": "mlx"})
        
        # Save metrics
        with open(save_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # upload to HF
        if self.push_to_hub:
            ### TODO
            raise NotImplementedError("Push to HF not implemented yet")
            upload_to_hub(save_path)
        
        # Manage checkpoint rotation
        if self.args.save_total_limit:
            ### TODO
            raise NotImplementedError("Checkpoint rotation not implemented yet")
            self._rotate_checkpoints()
    
    def _save_config(self):
        """Save training configuration."""
        config = {
            "model_type": self.model.__class__.__name__,
            "training_args": vars(self.args)
        }
        with open(self.output_dir / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

def upload_to_hub(
        path: str, 
        upload_repo: str, 
        hf_path: str,
        task_type: str,
        ):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
        task_type (str): Type of task the model was trained on.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    model_path = Path(path)

    card = ModelCard.load(hf_path) if ModelCard.exist_in_hub(hf_path) else ModelCard()
    card.data.tags = ["mlx" , "modernbert"] if card.data.tags is None else card.data.tags + ["mlx"] + ["modernbert"]
    card.data.base_model = hf_path
    card.data.task_type = task_type
    card.text = dedent(
        f"""
        # {upload_repo}

        The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was
        converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path})
        TODO

        # ModernBERT Model: {upload_repo}

        This model was trained using MLX ModernBERT for {task_type}.

        ## Usage

        ```python
        TODO
       
        ```
        """
    )
    # Save the model card
    card.save(model_path / "README.md")

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")