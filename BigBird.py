import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from  transformers.configuration_utils import PretrainedConfig
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

class BigBirdConfig(PretrainedConfig):
    model_type = "big_bird"

    def __init__(
        self,
        vocab_size=50358,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=4096,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        is_encoder_decoder=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sep_token_id=66,
        attention_type="block_sparse",
        use_bias=True,
        rescale_embeddings=False,
        block_size=64,
        num_random_blocks=3,
        gradient_checkpointing=False,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.is_encoder_decoder = is_encoder_decoder
        self.gradient_checkpointing = gradient_checkpointing

        self.rescale_embeddings = rescale_embeddings
        self.attention_type = attention_type
        self.use_bias = use_bias
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        
class BigBirdEmbeddings_with_Entity(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config,sub_embeds, obj_embeds ):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.sub_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.obj_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # End copy

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size
        self.sub_embeds=sub_embeds
        self.obj_embeds=obj_embeds

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,  past_key_values_length=0
    ):
        print('embedding done1!!!!!!!!!!!!!!!')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size ** 0.5)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        sub_embeddings = self.sub_embeddings(self.sub_embeds)
        obj_embeddings = self.obj_embeddings(self.obj_embeds)
        embeddings = inputs_embeds + token_type_embeddings + sub_embeddings + obj_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        print('embedding done1!!!!!!!!!!!!!!!')
        return embeddings

    