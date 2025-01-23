import argparse
import mlx.core as mx
from .utils import load, PIPELINES
from tuner.datasets import load_dataset, DatasetArgs
from tuner.trainer import Trainer, TrainingArgs

def main():

    model_path = "answerdotai/ModernBERT-base"
    dataset_id = "argilla/synthetic-domain-text-classification" # can be a local path
    task_type = "text-classification"
    is_regression = False # if true, it will be a regression task
    train = True # if false, it will only evaluate the model

    if task_type not in PIPELINES:
        raise ValueError(f"Task type {task_type} not supported. Choose from {PIPELINES.items()}")
        ### in practice, it would even be stopped at a later stage id training is not supported for the task type
    
    output_dir = model_path.split("/")[-1] + "_" + task_type

    # Load datasets
    dataset_args = DatasetArgs(
        data=dataset_id, 
        task_type=task_type, 
        train=train,
    )
    
    train_dataset, valid_dataset, test_dataset = load_dataset(dataset_args)

    model_config={}
    if task_type == "text-classification" and is_regression:
        model_config={"is_regression":True}
    if getattr(train_dataset,"id2label", None):
        model_config["id2label"] = train_dataset.id2label
        
    # Load model and tokenizer
    model, tokenizer = load(
        model_path, 
        model_config=model_config, 
        pipeline=task_type
    )

    # Training arguments
    training_args = TrainingArgs(
        batch_size=32,
        eval_batch_size=16,
        max_length= model.config.max_position_embeddings,
        num_train_epochs=3,
        learning_rate=2e-4, ### 5e-5 
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        eval_steps=500,
        save_steps=1000,
        logging_steps=1, ### 100
        output_dir=output_dir,
        save_total_limit=None,
        grad_checkpoint=True,
        push_to_hub=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        task_type=task_type,
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
