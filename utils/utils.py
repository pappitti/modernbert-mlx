# Copyright © 2023-2024 Apple Inc.

import contextlib
import copy
import glob
import importlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_reduce
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizer

# Local imports
from .tokenizer_utils import TokenizerWrapper, load_tokenizer
# Training imports 
from tuner.utils import nparams #, load_adapters ### removing LORA given it does not really make sense for small models

PIPELINES = [
    "embeddings",
    "masked-lm", 
    "text-classification", 
    "token-classification",
    "sentence-transformers",
    "zero-shot-classification",
    "sentence-similarity"
]

MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
}

MAX_FILE_SIZE_GB = 5

class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _get_classes(config: dict, pipeline: Optional[str] = 'masked-lm'):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    if pipeline not in PIPELINES:
        raise ValueError(f"Pipeline {pipeline} not supported. Supported pipelines: {PIPELINES}")

    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    if pipeline == "masked-lm":
        return arch.ModelForMaskedLM, arch.ModelArgs
    
    if pipeline == "text-classification":
        return arch.ModelForSequenceClassification, arch.ModelArgs
    
    if pipeline == "token-classification":
        return arch.ModelForTokenClassification, arch.ModelArgs
    
    if pipeline == "embeddings":
        return arch.Model, arch.ModelArgs
    
    if pipeline == "sentence-transformers":
        return arch.ModelForSentenceTransformers, arch.ModelArgs
    
    if pipeline == "zero-shot-classification":
        return arch.ModelForZeroShotClassification, arch.ModelArgs
    
    if pipeline == "sentence-similarity":
        return arch.ModelForSentenceSimilarity, arch.Model

    ### should not reach here
    return arch.Model, arch.ModelArgs

def _initialize_missing_classifier_weights(weights, model_args):
    """Initialize weights only if they don't exist in the weights dictionary."""
    # For sequence classification, check and initialize classifier weights if needed
    if getattr(model_args,"num_labels") is not None:
        if "classifier.weight" not in weights:
            initializer_range = getattr(model_args,"initializer_range", 0.02)
            weights["classifier.weight"] = mx.random.normal(
                (model_args.num_labels, model_args.hidden_size),
                scale=initializer_range
            )
            print(f"[INFO] Initialized classifier.weight with shape {weights['classifier.weight'].shape}")
            
        if "classifier.bias" not in weights:
            weights["classifier.bias"] = mx.zeros((model_args.num_labels,))

def compute_bits_per_weight(model):
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    model_params = sum(nparams(m) for _, m in leaf_modules)
    return model_bytes * 8 / model_params

def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        try:
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
            )
        except:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            ) from None
    return model_path


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config: dict = {},
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
    pipeline: Optional[str] = None,
) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        model_config (dict, optional): Configuration parameters for the model.
            Defaults to an empty dictionary.
        get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
            A function that returns the model class and model args class given a config.
            Defaults to the _get_classes function.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """

    config = load_config(model_path)
    config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files:
        # Try weight for back-compat
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))

    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = get_model_classes(config=config, pipeline=pipeline)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # Initialize any missing weights to train a base model based on the pipeline
    if pipeline == "text-classification":
        _initialize_missing_classifier_weights(weights, model_args)
    ### TODO add more pipelines

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may not have everything quantized
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model, config


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None, ## for now, disabling adapter loading
    lazy: bool = False,
    pipeline: Optional[str] = None,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model, config = load_model(model_path, lazy, model_config, pipeline=pipeline)
    ### disabling adapter for encoders
    # if adapter_path is not None:
    #     model = load_adapters(model, adapter_path)
    #     model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config)

    return model, tokenizer

def fetch_from_hub(
    model_path: Path, lazy: bool = False
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model, config = load_model(model_path, lazy)
    tokenizer = load_tokenizer(
        model_path, eos_token_ids=config.get("eos_token_id", None)
    )
    return model, config, tokenizer


# def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
#     """
#     Splits the weights into smaller shards.

#     Args:
#         weights (dict): Model weights.
#         max_file_size_gb (int): Maximum size of each shard in gigabytes.

#     Returns:
#         list: List of weight shards.
#     """
#     max_file_size_bytes = max_file_size_gb << 30
#     shards = []
#     shard, shard_size = {}, 0
#     for k, v in weights.items():
#         if shard_size + v.nbytes > max_file_size_bytes:
#             shards.append(shard)
#             shard, shard_size = {}, 0
#         shard[k] = v
#         shard_size += v.nbytes
#     shards.append(shard)
#     return shards


# def save_weights( ### handled in the trainer
#     save_path: Union[str, Path],
#     weights: Dict[str, Any],
#     *,
#     donate_weights: bool = False,
# ) -> None:
#     """Save model weights into specified directory."""
#     if isinstance(save_path, str):
#         save_path = Path(save_path)
#     save_path.mkdir(parents=True, exist_ok=True)

#     shards = make_shards(weights)
#     shards_count = len(shards)
#     shard_file_format = (
#         "model-{:05d}-of-{:05d}.safetensors"
#         if shards_count > 1
#         else "model.safetensors"
#     )

#     total_size = sum(v.nbytes for v in weights.values())
#     index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

#     # Write the weights and make sure no references are kept other than the
#     # necessary ones
#     if donate_weights:
#         weights.clear()
#         del weights

#     for i in range(len(shards)):
#         shard = shards[i]
#         shards[i] = None
#         shard_name = shard_file_format.format(i + 1, shards_count)
#         shard_path = save_path / shard_name

#         mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

#         for weight_name in shard.keys():
#             index_data["weight_map"][weight_name] = shard_name
#         del shard

#     index_data["weight_map"] = {
#         k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
#     }

#     with open(save_path / "model.safetensors.index.json", "w") as f:
#         json.dump(
#             index_data,
#             f,
#             indent=4,
#         )


def quantize_model(
    model: nn.Module,
    config: dict,
    q_group_size: int = 64,
    q_bits: int = 4,
    quant_predicate: Optional[
        Callable[[str, nn.Module, dict], Union[bool, dict]]
    ] = None,
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.
        quant_predicate (Callable): A callable that decides how
            to quantize each layer based on the path.
            Accepts the layer `path`, the `module` and the model `config`.
            Returns either a bool to signify quantize/no quantize or
            a dict of quantization parameters to pass to `to_quantized`.

    Returns:
        Tuple: Tuple containing quantized weights and config.
    """
    quantized_config = copy.deepcopy(config)
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}

    # Add any custom quantization parameters to the config as we go
    def _class_predicate(p, m):
        bool_or_params = quant_predicate(p, m, config)
        quantized_config["quantization"][p] = bool_or_params
        return bool_or_params

    nn.quantize(
        model,
        q_group_size,
        q_bits,
        class_predicate=_class_predicate if quant_predicate else None,
    )
    # support hf model tree #957
    quantized_config["quantization_config"] = quantized_config["quantization"]
    quantized_weights = dict(tree_flatten(model.parameters()))

    bpw = compute_bits_per_weight(model)
    print(f"[INFO] Quantized model with {bpw:.3f} bits per weight.")

    return quantized_weights, quantized_config


# def save_config( ### handled in the trainer
#     config: dict,
#     config_path: Union[str, Path],
# ) -> None:
#     """Save the model configuration to the ``config_path``.

#     The final configuration will be sorted before saving for better readability.

#     Args:
#         config (dict): The model configuration.
#         config_path (Union[str, Path]): Model configuration file path.
#     """
#     # Clean unused keys
#     config.pop("_name_or_path", None)

#     # sort the config for better readability
#     config = dict(sorted(config.items()))

#     # write the updated config to the config_path (if provided)
#     with open(config_path, "w") as fid:
#         json.dump(config, fid, indent=4)


# def convert(
#     hf_path: str,
#     mlx_path: str = "mlx_model",
#     quantize: bool = False,
#     q_group_size: int = 64,
#     q_bits: int = 4,
#     dtype: str = "float16",
#     upload_repo: str = None,
#     revision: Optional[str] = None,
#     dequantize: bool = False,
#     quant_predicate: Optional[
#         Callable[[str, nn.Module, dict], Union[bool, dict]]
#     ] = None,
# ):
#     # Check the save path is empty
#     if isinstance(mlx_path, str):
#         mlx_path = Path(mlx_path)

#     if mlx_path.exists():
#         raise ValueError(
#             f"Cannot save to the path {mlx_path} as it already exists."
#             " Please delete the file/directory or specify a new path to save to."
#         )

#     print("[INFO] Loading")
#     model_path = get_model_path(hf_path, revision=revision)
#     model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

#     weights = dict(tree_flatten(model.parameters()))
#     dtype = getattr(mx, dtype)
#     weights = {k: v.astype(dtype) for k, v in weights.items()}

#     if quantize and dequantize:
#         raise ValueError("Choose either quantize or dequantize, not both.")

#     if quantize:
#         print("[INFO] Quantizing")
#         model.load_weights(list(weights.items()))
#         weights, config = quantize_model(
#             model, config, q_group_size, q_bits, quant_predicate=quant_predicate
#         )

#     if dequantize:
#         print("[INFO] Dequantizing")
#         model = dequantize_model(model)
#         weights = dict(tree_flatten(model.parameters()))

#     del model
#     save_weights(mlx_path, weights, donate_weights=True)

#     py_files = glob.glob(str(model_path / "*.py"))
#     for file in py_files:
#         shutil.copy(file, mlx_path)

#     tokenizer.save_pretrained(mlx_path)

#     save_config(config, config_path=mlx_path / "config.json")

#     if upload_repo is not None:
#         upload_to_hub(mlx_path, upload_repo, hf_path)