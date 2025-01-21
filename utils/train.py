import argparse
import mlx.core as mx
from .utils import load, PIPELINES
from tuner.datasets import load_dataset
from tuner.trainer import Trainer, TrainingArgs

def main():

    model_path = "answerdotai/ModernBERT-base"
    dataset_id = "argilla/synthetic-domain-text-classification"
    task_type = "text-classification"
    is_regression = False # if true, it will be a regression task
    train = True # if false, it will only evaluate the model

    if task_type not in PIPELINES:
        raise ValueError(f"Task type {task_type} not supported. Choose from {PIPELINES.items()}")
        ### in practice, it would even be stopped at a later stage id training is not supported for the task type
    
    output_dir = model_path.split("/")[-1] + "_" + task_type

    # Load datasets
    train_dataset, valid_dataset, test_dataset = load_dataset(dataset_id, task_type)

    ### TODO: make sure that labels are ids not text. if not, convert them to using label2id
    ## update id2label and label2id in the model.config, update num_labels accordingly
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    if task_type == "text-classification" and is_regression:
        model_config={"is_regression":True}
    else:
        model_config={}
    
    # Load model and tokenizer
    model, tokenizer = load(
        model_path, 
        model_config=model_config, 
        pipeline=task_type
    )

    # Training arguments
    training_args : TrainingArgs = {
        "batch_size": 32,
        "eval_batch_size": 16,
        "max_length": model.config.max_position_embeddings,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "eval_steps": 500,
        "save_steps": 1000,
        "logging_steps": 100,
        "output_dir": output_dir,
        "save_total_limit": None,
        "grad_checkpoint": True,
        "push_to_hub": False,
    }

    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )
    
    # Train or evaluate
    if train:
        trainer.train()
    if test_dataset:
        trainer.test(test_dataset)

if __name__ == "__main__":
    main()
