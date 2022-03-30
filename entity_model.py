import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig,BigBirdModel, BigBirdPreTrainedModel, RobertaPreTrainedModel
from embeddings import *
import torch.nn.functional as F
from transformers.activations import ACT2FN
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
    
    def entity_features(self,features,sub_mask,obj_mask):
        entity=[]
        for i in range(len(features)):
            sub_start = sub_mask[i].tolist().index(7)
            sub_end = sub_mask[i].tolist().index(8)
            obj_start = obj_mask[i].tolist().index(9)
            obj_end = obj_mask[i].tolist().index(10)
            total = (features[i,sub_start,:]*2+features[i,sub_end,:]*2+features[i,obj_start,:]+features[i,obj_end,:])/6
            entity.append(total.tolist())
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        entity = torch.tensor(entity).to(device)
        return entity
    
    def forward(self, features, sub_mask,obj_mask, **kwargs):
        x = self.entity_features(features,sub_mask,obj_mask)
          # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
#class Entity_Embedding_Model(RobertaPreTrainedModel):
class Entity_Embedding_Model(BigBirdPreTrainedModel):
    def __init__(self, config, dropout_rate):
        super(Entity_Embedding_Model, self).__init__(config)
        self.model = AutoModel.from_pretrained('monologg/kobigbird-bert-base')
        self.model_config = config
        self.model.embeddings =Entity_Embeddings(config)
        self.model_config.num_labels = 30
        self.num_labels = 30

        #self.cls_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, dropout_rate)
        #self.entity_fc_layer1 = FCLayer(self.config.hidden_size, self.config.hidden_size, dropout_rate)
        #self.entity_fc_layer2 = FCLayer(self.config.hidden_size, self.config.hidden_size, dropout_rate)

        #self.label_classifier = FCLayer(
        #    self.config.hidden_size * 3,
        #    self.config.num_labels,
        #    dropout_rate,
        #    use_activation=False
        #)
        self.classifier = ClassificationHead(config)


    def forward(self, input_ids, attention_mask,token_type_ids, sub_mask, obj_mask, labels):
        outputs = self.model(
            input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state
        
        logits = self.classifier(sequence_output,sub_mask,obj_mask)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

 