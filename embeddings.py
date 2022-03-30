import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import transformers
from  transformers.configuration_utils import PretrainedConfig
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from packaging import version

class Entity_Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        self.sub_embeddings= nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.obj_embeddings= nn.Embedding(config.type_vocab_size, config.hidden_size)
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )
        # End copy

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        
        device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if input_ids is not None:
            input_shape = input_ids.size()
            sub_embeds=[]
            obj_embeds=[]
            for idx, id in enumerate(input_ids):
                id=id.tolist()
                sub=[0]*len(id)
                obj=[0]*len(id)
                sub_start = id.index(7)
                sub_end = id.index(8)
                obj_start = id.index(9)
                obj_end = id.index(10)
                sub[sub_start+1:sub_end]=[1]*(sub_end-sub_start-1)
                obj[obj_start+1:obj_end]=[1]*(obj_end-obj_start-1)
                sub_embeds.append(sub)
                obj_embeds.append(obj)
            sub_embeds=torch.tensor(sub_embeds).to(device)
            obj_embeds=torch.tensor(obj_embeds).to(device)

        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size ** 0.5)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        sub_embeddings = self.sub_embeddings(sub_embeds)
        obj_embeddings = self.obj_embeddings(obj_embeds)
        
        embeddings = inputs_embeds + token_type_embeddings + sub_embeddings+obj_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings

    