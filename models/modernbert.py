import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union, Literal, Any

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


### NOTE : removed all the attention_outputs (eager mode), may add id back later
### given no flash attention 2, padded/unpadded was also removed ?
### TODO: 
## general model
# checkpoint


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q : The query tensor.
        k : The key tensor.
        cos : The cosine part of the rotary embedding.
        sin : The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    ### hidden_activation : str = "gelu" ## should not be necessary, we'd just apply gelu.
    max_position_embeddings: Optional[int] = None
    norm_eps: float = 1e-05,
    norm_bias : bool = False,
    global_rope_theta : float = 160000.0,
    attention_bias: bool = False,
    attention_dropout : float =0.0,
    global_attn_every_n_layers : int =3,
    local_attention : int =128,
    local_rope_theta: float = 10000 ,
    embedding_dropout : float =0.0,
    mlp_bias: bool = False,
    mlp_dropout : float = 0.0 ,

    initializer_range=0.02, # relevant for MLX?
    initializer_cutoff_factor=2.0, # relevant for MLX?
    pad_token_id=50283, ## relevant?
    eos_token_id=50282,
    bos_token_id=50281,
    cls_token_id=50281,
    sep_token_id=50282,
    decoder_bias=True,
    classifier_pooling: Literal["cls", "mean"] = "cls",
    classifier_dropout=0.0, 
    classifier_bias=False,
    # classifier_activation="gelu"
    # deterministic_flash_attn=False ## for torch only, to remove???
    sparse_prediction=False 
    sparse_pred_ignore_index=-100 
    # reference_compile=None ## for torch only, to remove???

    # output_attentions: bool = False # not relevant if we only use sdpa
    output_hidden_states: bool = False 
    use_return_dict: bool = True 


class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int =2048, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
    
    ### property decorator to make a method behave like an attribute and avoid a flag for missing parameters
    ### the flip side is that the value is recalculated at every forward pass 
    ### TBC for training
    @property
    def inv_freq(self):
        return 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=mx.int32) / self.dim))

    def __call__(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = mx.expand_dims(self.inv_freq, [0, 2])
        inv_freq_expanded = mx.broadcast_to(
            inv_freq_expanded,
            [position_ids.shape[0], inv_freq_expanded.shape[1], 1]
        )

        position_ids_expanded = mx.expand_dims(position_ids.astype(mx.float32), 1)
    
        # Computing position embeddings
        freqs = mx.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = mx.transpose(freqs, [0, 2, 1])
        
        # Duplicating frequencies
        emb = mx.concatenate([freqs, freqs], axis=-1)
        
        # Computing sin and cos
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        
        return cos.astype(x.dtype), sin.astype(x.dtype)


class ModernBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config: ModelArgs):
        super().__init__() ## need this?
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias) 
        self.drop = nn.Dropout(p=config.embedding_dropout)

    def __call__(self, input_ids):
        embeddings = self.tok_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings

class ModernBertMLP(nn.Module):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces class BertIntermediate`
    and class SelfOutput with a single module that has similar functionality.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size *2, bias=config.mlp_bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=config.mlp_dropout)
        self.Wo = nn.Linear(int(config.intermediate_size), config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states):
        x = self.Wi(hidden_states)
        # Implementing chunk operation
        split_dim = x.shape[-1] // 2
        input, gate = x[:, :, :split_dim], x[:, :, split_dim:] ### I need to understand this better : https://arxiv.org/pdf/2002.05202v1
        return self.Wo(self.drop(self.act(input) * gate))

class ModernBertAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.
    For now, only supports the Scaled Dot-Product Attention (SDPA) implementation.
    """
    def __init__(self, config: ModelArgs, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads ({config.num_attention_heads})"
            )
        
        self.attention_dropout = config.attention_dropout
        # self.deterministic_flash_attn = config.deterministic_flash_attn ### for torch only, to remove???
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        max_position_embeddings = config.max_position_embeddings
        if self.local_attention != (-1, -1):
            if config.local_rope_theta is not None:
                rope_theta = config.local_rope_theta
            max_position_embeddings = config.local_attention

        self.rotary_emb = ModernBertRotaryEmbedding(
            dim=self.head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta
        )

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(p=config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()
        
    def __call__(
            self, 
            hidden_states, 
            attention_mask = None,
            sliding_window_mask = None,
            position_ids=None,
            # output_attentions: Optional[bool] = False, ### is not used with sdpa (only with flash attention 2 and eager mode),
            **kwargs
        ):
        qkv = self.Wqkv(hidden_states)
        bs = hidden_states.shape[0]
        qkv = mx.reshape(qkv, (bs, -1, 3, self.num_heads, self.head_dim))

        # Get attention outputs using SDPA
        cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
        qkv = mx.transpose(qkv, [0, 3, 2, 1, 4])  # [batch_size, nheads, 3, seqlen, headdim]
        query, key, value = mx.split(qkv, indices_or_sections=3, axis=2)
        query = query.squeeze(2) 
        key = key.squeeze(2)
        value = value.squeeze(2)

        # Applying rotary embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Handling local attention if needed
        if self.local_attention != (-1, -1):
            attention_mask = sliding_window_mask

        # Computing attention using MLX's SDPA
        scale = 1.0 / math.sqrt(query.shape[-1])
        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value,
            scale=scale,
            mask=attention_mask
        )
        
        # Reshaping and apply output projection
        attn_output = mx.transpose(attn_output, [0, 2, 1, 3])
        attn_output = mx.reshape(attn_output, (bs, -1, self.all_head_size))
        
        # Applying output projection and dropout
        hidden_states = self.Wo(attn_output)
        hidden_states = self.out_drop(hidden_states)

        return (hidden_states,)


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp = ModernBertMLP(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def __call__(
            self, 
            hidden_states , 
            attention_mask =None, 
            sliding_window_mask = None,
            position_ids  = None,
            # output_attentions: Optional[bool] = False, ## should not be used with sdpa
    ):
        normalized_hidden_states = self.attn_norm(hidden_states)
        attention_output = self.attn( 
            normalized_hidden_states, 
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            # output_attentions=output_attentions
        )
        hidden_states = hidden_states + attention_output[0]
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_output

        return (hidden_states,)   # removed attention outputs


class ModernBertModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [
            ModernBertEncoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False ### TBC

    def get_input_embeddings(self) -> ModernBertEmbeddings:
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def __call__(
            self, 
            input_ids, 
            attention_mask = None, # shape: (batch_size, seq_len) see below
            sliding_window_mask = None,
            position_ids = None,
            # output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ):
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions # should not be used with sdpa
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = () if output_hidden_states else None
        # all_attentions = () if output_attentions else None # should not be used with sdpa

        # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask) ### work out the padding before removing this line

        batch_size, seq_len = input_ids.shape[:2]

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len)) ### not sure why transformers decided to do this and then _update_attention_mask() below

        if position_ids is None:
            position_ids = mx.arange(seq_len, dtype=mx.int32)[None, :]

        # get attention mask and sliding window mask
        attention_mask, sliding_window_mask = self._update_attention_mask(
            attention_mask=attention_mask,
            # output_attentions=False ### should not be used with sdpa
        )

        hidden_states = self.embeddings(input_ids)

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # if self.gradient_checkpointing and self.training:
            #     ### TODO ?
            #     layer_outputs = mx.checkpoint(
            #         encoder_layer.__call__,
            #         hidden_states,
            #         attention_mask,
            #         sliding_window_mask,
            #         position_ids,
            #         # output_attentions,
            #     )
            # else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                # output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
        
        hidden_states = self.final_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            # "attentions": all_attentions,
        }
    
    def _update_attention_mask(self, attention_mask): ### move to base.py ??
        
        # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        attention_mask = attention_mask[:, None, None, :]
        
        # Create the causal mask for global attention
        # (1, 1, seq_len, seq_len)
        seq_len = attention_mask.shape[-1]
        global_attention_mask = mx.broadcast_to(attention_mask, (attention_mask.shape[0], 1, seq_len, seq_len))
        
        # Create position indices for sliding window
        rows = mx.arange(seq_len)
        rows = rows[None, :]  # (1, seq_len)
        # Calculate position-wise distances
        distance = mx.abs(rows - rows.T)  # (seq_len, seq_len)
        
        # Create sliding window mask using mx.where
        window_mask = mx.where(
            distance <= (self.config.local_attention // 2),
            mx.ones_like(distance),
            mx.zeros_like(distance)
        )
        
        # Expand dimensions using None indexing
        window_mask = window_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
            
        # Broadcast to match batch size
        window_mask = mx.broadcast_to(window_mask, global_attention_mask.shape)
        
        # Creating sliding window attention mask
        # Replacing non-window positions with large negative value
        sliding_window_mask = mx.where(
            window_mask,
            global_attention_mask,
            float('-inf') ## if not broadcasted for some reason : float('-inf') * mx.ones_like(global_attention_mask)
        )
    
        return global_attention_mask, sliding_window_mask


### for embeddings 
## this is a hack to align with other models here while downloading weights with the maskedlm config from HF
## the decoder.bias is ignored in the model
class Model(nn.Module): 
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)

    def __call__(
        self, 
        input_ids, 
        attention_mask: Optional[mx.array] = None,
    ):
        
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape 
            attention_mask = mx.ones((batch_size, seq_len)) ### not sure why transformers decided to do this but it is updated via _update_attention_mask() in the model

        # Get embeddings and encoder outputs as before
        encoder_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = encoder_outputs["last_hidden_state"] if isinstance(encoder_outputs, dict) else encoder_outputs[0] 
        
        # Do pooling here (unlike BERT)
        if self.config.classifier_pooling == "cls":
            pooled = sequence_output[:, 0]
        elif self.config.classifier_pooling == "mean":                
            attention_mask = mx.expand_dims(attention_mask, -1)
            pooled = mx.sum(sequence_output * attention_mask, axis=1) / mx.sum(attention_mask, axis=1)
            
        # normalization
        pooled = pooled / mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True) + 1e-12)

        print("sequence_output", sequence_output.shape)

        return sequence_output, pooled 
    
    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                ### used this from another model, may need to change
                # Remove unused position_ids
                continue
            if k in ["head.norm.weight", "head.dense.weight", "decoder.bias"]:
                ### this is the hack
                continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights

### below are the classes for specific usecases
class ModernBertPredictionHead(nn.Module):
    def __init__(self, config : ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False) ### current HF checkpoint does not have bias for the dense layer
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
class ModelForMaskedLM(nn.Module):
    def __init__(self, config : ModelArgs):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config) ## no bias for this in the current HF checkpoint
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        # Tie weights ### does not seem to work (sanitizing the weights to enforce weight tying)
        self.tie_weights()
    
    def tie_weights(self):
        embedding_layer = self.model.get_input_embeddings()
        self.decoder.weight = embedding_layer.weight
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def get_output_embeddings(self):
        return self.decoder
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        self.tie_weights()  # Re-tie weights after setting new embeddings
    
    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings
        self.tie_weights()  # Re-tie weights after setting new decoder
        
    def __call__(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Dict:
        
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape 
            attention_mask = mx.ones((batch_size, seq_len)) ### not sure why transformers decided to do this but it is updated via _update_attention_mask() in the model

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs["last_hidden_state"] if return_dict else outputs[0]
        logits = self.head(hidden_states)  
        logits = self.decoder(logits)
        
        loss = None
        if labels is not None :  
            ### TBC
            if getattr(self.config, "sparse_prediction", False):
                # Flatten labels and predictions
                flat_labels = labels.reshape(-1)
                flat_predictions = logits.reshape(-1, logits.shape[-1])
                
                # Filter out non-masked tokens
                ignore_index = getattr(self.config, "sparse_pred_ignore_index", -100)
                mask_tokens = flat_labels != ignore_index
                
                # Only compute loss on masked tokens
                masked_predictions = flat_predictions[mask_tokens]
                masked_labels = flat_labels[mask_tokens]
                
                loss = nn.losses.cross_entropy(
                    masked_predictions,
                    masked_labels,
                    reduction='mean'
                )
            else:
                # Standard loss computation on all tokens
                loss = nn.losses.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction='mean'
                )
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states", None),
        }
    
    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            if k == "model.embeddings.tok_embeddings.weight":
                ### going around the weight tying issue. TODO : improve this
                sanitized_weights["decoder.weight"] = v
                sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v
        return sanitized_weights
    
class ModelForSequenceClassification(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = nn.Dropout(p=config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=config.classifier_bias)

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Dict:
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones((batch_size, seq_len))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs["last_hidden_state"] if return_dict else outputs[0]

        # Pooling strategy
        if self.config.classifier_pooling == "cls":
            pooled = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            attention_mask = mx.expand_dims(attention_mask, -1)
            pooled = mx.sum(last_hidden_state * attention_mask, axis=1) / mx.sum(attention_mask, axis=1)

        # Apply head, dropout and classifier
        pooled = self.head(pooled)
        pooled = self.drop(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None :
            if self.num_labels == 1:
                # Regression
                loss = nn.losses.mse_loss(logits.squeeze(), labels.squeeze())
            else:
                if len(labels.shape) == 1 or labels.shape[-1] == 1:
                    # Single-label classification
                    loss = nn.losses.cross_entropy(
                        logits.reshape(-1, self.num_labels),
                        labels.reshape(-1)
                    )
                else:
                    # Multi-label classification
                    loss = nn.losses.binary_cross_entropy(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states", None),
        }
    
    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            # if k in ["head.dense.bias"]:  # Add any other weights that should be skipped
            #     continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights
    

class ModelForTokenClassification(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = nn.Dropout(p=config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=config.classifier_bias)

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Dict:
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones((batch_size, seq_len))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs["last_hidden_state"] if return_dict else outputs[0]
        
        # Apply prediction head, dropout, and classification layer to each token
        sequence_output = self.head(sequence_output)
        sequence_output = self.drop(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Compute token classification loss
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1)
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states", None),
        }
    
    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            # if k in ["head.dense.bias"]:  # Add any other weights that should be skipped
            #     continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights