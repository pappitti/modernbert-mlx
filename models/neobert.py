### This is a implementation of  the NeoBERT model, based on this repository:
### https://github.com/chandar-lab/NeoBERT/blob/main/src/neobert/model/model.py
### not all features are implemented, only the base model
### notably NormNeoBert and NormEncoderBlock are not implemented

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, List

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    BaseModelOutput,
    mean_pooling,
    normalize_embeddings,
)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 768
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0
    embedding_init_range: float = 0.02
    decoder_init_range: float = 0.02
    rms_norm: bool = True
    rope: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-06
    hidden_act: str = "SwiGLU"
    vocab_size: int = 32064
    pad_token_id: int = 0
    max_length: int = 1024
    base_scale: float = 1.0 / (960.0**0.5) # Only for NormNeoBERT (not implemented, placeholder)
    ngpt: bool = False
    architectures: List[str] = field(
        default_factory=lambda: ["NeoBERTLMHead"]
    )

    # pipeline args, mostly for classification
    classifier_pooling: Literal["cls", "mean"] = "mean"
    is_regression: Optional[bool] = None  # for Sequence Classification
    label2id: Optional[Dict[str, int]] = None  # for Sequence Classification
    id2label: Optional[Dict[int, str]] = None  # for Sequence Classification
    classifier_dropout: float = 0.1
    classifier_init_range: float = 0.02

    @property
    def dim_head(self) -> int:
        """Dimension of each attention head."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_labels(self) -> int:  # for Sequence Classification
        """
        Number of labels is determined by:
        - For zero-shot classification: length of label_candidates
        - For regression or binary with sigmoid: 1
        - For classification: length of id2label mapping
        """

        if self.is_regression:
            return 1

        if self.pipeline_config and self.pipeline_config.get("binary_sigmoid", False):
            return 1

        if self.id2label is None:
            raise ValueError(
                "id2label mapping must be provided for categorical classification. "
                "For regression or binary classification with sigmoid output, "
                "set is_regression=True or binary_sigmoid=True in pipeline_config."
            )

        return len(self.id2label)

# Can't use nn.RoPE here bacause of the shape of freqs_cis so we implement it manually
def precompute_freqs_cis(dim: int, end: int, theta: float) -> mx.array:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(end, dtype=mx.float32)
    freqs = mx.outer(t, freqs)

    freqs_cos = mx.cos(freqs)
    freqs_sin = mx.sin(freqs)
    return (freqs_cos, freqs_sin)


def apply_rotary_emb(
    x: mx.array,
    freqs: tuple[mx.array, mx.array],
) -> tuple[mx.array, mx.array]:
    
    freqs_cos, freqs_sin = freqs

    x_r = x.reshape(*x.shape[:-1], -1, 2)
    
    # Split into the two parts for rotation
    x1 = x_r[..., 0]
    x2 = x_r[..., 1]
    
    # freqs have shape (L, D_h/2)
    freqs_cos = freqs_cos.reshape(1, *freqs_cos.shape)
    freqs_cos = freqs_cos.reshape(1, freqs_cos.shape[1], 1, freqs_cos.shape[2])
    freqs_sin = freqs_sin.reshape(1, *freqs_sin.shape)
    freqs_sin = freqs_sin.reshape(1, freqs_sin.shape[1], 1, freqs_sin.shape[2])

    # Perform the rotation
    x_out1 = x1 * freqs_cos - x2 * freqs_sin
    x_out2 = x1 * freqs_sin + x2 * freqs_cos
    
    # Re-assemble the rotated tensor
    x_rotated = mx.stack([x_out1, x_out2], axis=-1)
    
    # Flatten the last two dimensions to restore the original shape
    return x_rotated.reshape(*x.shape)


class SwiGLU(nn.Module):
    """TBC when model weights are available"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        # The first linear layer projects from the input dimension to twice the hidden
        # dimension to create both the gate and the up-projection in one pass.
        self.w12 = nn.Linear(input_dim, 2 * hidden_dim, bias=bias)
        
        self.w3 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.w12(x)

        # Split the tensor into two halves along the last dimension
        gate, up = mx.split(gate_up, 2, axis=-1)
        fused_output = nn.silu(gate) * up

        return self.w3(fused_output)


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config

        # Attention
        self.qkv = nn.Linear(
            config.hidden_size, config.hidden_size*3, bias=False
        )
        # self.rotary_emb = nn.RoPE(dims=config.dim_head, base=config.rope_theta)
        self.scale = config.dim_head**-0.5 # necessary for scaled dot-product attention in MLX unlike PyTorch
        self.wo=nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.resid_dropout = nn.Dropout(p=config.dropout)

        # Feedforward network
        match config.hidden_act.lower():
            case "swiglu":
                # To keep the number of parameters and the amount of computation constant, we reduce the number of
                # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
                # avoid RuntimeError due to misaligned operand
                multiple_of = 8
                intermediate_size = int(2 * config.intermediate_size / 3)
                intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
                self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=False)
            case "gelu":
                self.ffn = nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                    nn.GELU(),
                    nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                )

        self.attention_norm = (
            nn.RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )
        self.ffn_norm = (
            nn.RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        self.ffn_dropout = nn.Dropout(config.dropout)

    def __call__(
            self, 
            hidden_states: mx.array, 
            attention_mask: Optional[mx.array] = None,
            freqs: Optional[mx.array] = None
        ) -> mx.array:
        hidden_states = hidden_states + self._att_block(self.attention_norm(hidden_states), attention_mask, freqs=freqs)[0]
        hidden_states = hidden_states + self._ff_block(self.ffn_norm(hidden_states))[0]
        return (hidden_states,)

    def _att_block(
        self, 
        hidden_states: mx.array, 
        attention_mask: Optional[mx.array] = None,
        freqs: Optional[mx.array] = None
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        qkv_x = self.qkv(hidden_states) # (B, L, 3 * num_attention_heads * dim_head)
        qkv_x = mx.reshape(qkv_x, (B, L, self.config.num_attention_heads, self.config.dim_head*3)) # (B, L, num_attention_heads, dim_head * 3)
        # qkv_x = mx.transpose(qkv_x, (0, 2, 1, 3))  # (B, num_attention_heads, L, dim_head * 3)
        # TODO : check transpose above (different from original repo but may be due to how rope is implemented in mlx)
        queries, keys, values = mx.split(qkv_x, 3, axis=-1)  # (B, num_attention_heads, L, dim_head) 
  
        # Applying rotary embeddings
        # queries = self.rotary_emb(queries)
        # keys = self.rotary_emb(keys)
        if self.config.rope and freqs is not None:
            queries = apply_rotary_emb(queries, freqs)
            keys = apply_rotary_emb(keys, freqs)

        # Transpose for attention AFTER RoPE
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, 
            keys, 
            values, 
            scale=self.scale,
            mask=attention_mask
            # dropout_p=self.config.dropout_prob if self.training else 0.0,
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        hidden_states = self.wo(output)
        hidden_states = self.resid_dropout(hidden_states)

        return (hidden_states,)
    
    def _ff_block(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)
        return (hidden_states,)


class NeoBERTModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size)

        if self.config.rope:
            self.freqs = precompute_freqs_cis(
                config.dim_head, config.max_length, config.rope_theta
            )

        self.transformer_encoder = [
            EncoderBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.layer_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)

    def _update_attention_mask(self, attention_mask: Optional[mx.array] = None):
        """
        Creates a causal mask and combines it with the padding mask.
        """
        dtype = attention_mask.dtype
        B, L = attention_mask.shape

        # Reshape padding mask from (B, L) to (B, 1, 1, L) to be broadcastable
        padding_mask = attention_mask[:, None, None, :]
        additive_padding_mask = mx.where(padding_mask == 1, 0.0, -1e9).astype(dtype)

        return additive_padding_mask

    def __call__(
            self, input_ids: mx.array, 
            attention_mask: mx.array = None,
            position_ids: Optional[mx.array] = None
    ):
        attention_mask = self._update_attention_mask(
            attention_mask,
        )

        hidden_states = self.encoder(input_ids)

        freqs = (
            self.freqs[0][: input_ids.shape[1]] if self.config.rope else None,
            self.freqs[1][: input_ids.shape[1]] if self.config.rope else None,
        )

        for layer in self.transformer_encoder:
            layer_outputs = layer(hidden_states, attention_mask, freqs=freqs)
            hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
        }


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = NeoBERTModel(config)

        if config.architectures == ["NeoBERTLMHead"]:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        elif config.architectures == ["NeoBERTForSequenceClassification"]:
            self.num_labels = config.num_labels
            self.is_regression = config.is_regression
            self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.dropout = nn.Dropout(p=config.classifier_dropout)
            self.classifier = nn.Linear(config.hidden_size,config.num_labels)

    def _process_outputs(self, logits: mx.array) -> mx.array:
        """Apply the appropriate activation function to the logits for classification tasks."""
        if self.is_regression:
            return logits  # No activation for regression
        elif self.num_labels == 1:
            return mx.sigmoid(logits)  # Binary classification
        else:
            # Using softmax for multi-class classification
            return mx.softmax(logits, axis=-1)

    def __call__(
            self, 
            input_ids: mx.array, 
            attention_mask: mx.array = None,
            position_ids: Optional[mx.array] = None,
            return_dict: Optional[bool] = False,
    ) -> BaseModelOutput:
        
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.encoder.weight.dtype,
            )

        out = self.model(
            input_ids, 
            attention_mask,
            position_ids=position_ids
        )

        last_hidden_state = (
            out["last_hidden_state"] if isinstance(out, dict) else out[0]
        )

        # Pooling strategy using config
        if self.config.classifier_pooling == "cls":
            pooled_embeddings = last_hidden_state[:, 0]
        else:
            # Fallback to last token pooling
            pooled_embeddings = mean_pooling(last_hidden_state, attention_mask)

        text_embeds = normalize_embeddings(pooled_embeddings)

        pooled_output = None
        # placeholder for masked LM and sequence classification heads
        if self.config.architectures == ["NeoBERTLMHead"]:
            pooled_output = self.decoder(last_hidden_state)
        elif self.config.architectures == ["NeoBERTForSequenceClassification"]:
            pooled_output = self.dropout(pooled_embeddings)
            pooled_output = self.dense(pooled_output)
            pooled_output = nn.tanh(pooled_output)
            pooled_output = self.dropout(pooled_output)
            pooled_output = self.classifier(pooled_output)
            pooled_output = self._process_outputs(pooled_output)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            text_embeds=text_embeds,
            pooler_output=pooled_output,
        )

    # TODO
    def sanitize(self, weights):
        # no need for lm_head.weight in Qwen3 for embedding models
        sanitized_weights = {}
        for k, v in weights.items():
            # # Filter out the language model head, which is not used for embeddings.
            # if "lm_head.weight" in k:
            #     continue

            # new_key = f"model.{k}"
            sanitized_weights[k] = v
        return sanitized_weights
