import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BigBirdModel, RobertaPreTrainedModel, BigBirdPreTrainedModel
import transformers


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class R_BigBird(RobertaPreTrainedModel):
    def __init__(self, config, dropout_rate):
        super(R_BigBird, self).__init__(config)
        self.model = AutoModel.from_pretrained('klue/roberta-large')
        self.model_config = config
        self.model_config.num_labels = 30
        self.num_labels = 30


        self.hidden_dim = self.model_config.hidden_size
        self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 2, dropout= 0.2,
                            batch_first= True, bidirectional= True)
        self.fc= nn.Linear(self.hidden_dim*2, self.model_config.num_labels)






        self.cls_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, dropout_rate)
        self.entity_fc_layer1 = FCLayer(self.config.hidden_size, self.config.hidden_size, dropout_rate)
        self.entity_fc_layer2 = FCLayer(self.config.hidden_size, self.config.hidden_size, dropout_rate)

        #self.label_classifier = FCLayer(
        #    self.config.hidden_size * 3,
        #    self.config.num_labels,
        #    dropout_rate,
        #    use_activation=False
        #)
        self.label_classifier = FCLayer(
            self.config.hidden_size * 2,
            self.config.num_labels,
            dropout_rate,
            use_activation=False
        )
    @staticmethod
    def entity_average(hidden_output, e_mask):
        e_mask_unsqueeze = e_mask.unsqueeze(1)
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()
        return avg_vector

    def forward(self, input_ids, attention_mask,token_type_ids, sub_mask, obj_mask, labels):
        outputs = self.model(
            input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state

        e1_h = self.entity_average(sequence_output, sub_mask)
        e2_h = self.entity_average(sequence_output, obj_mask)

        sentence_representation=outputs.pooler_output


        #hidden, (last_hidden, last_cell)= self.lstm(sequence_output)
        #cat_hidden= torch.cat((last_hidden[0], last_hidden[1]), dim= 1)
        #logits= self.fc(cat_hidden)

        mask=sub_mask+obj_mask
        sequence_output=sequence_output[mask !=0,:].view(32,-1,self.config.hidden_size)
        hidden, (last_hidden, last_cell)= self.lstm(sequence_output)
        cat_hidden= torch.cat((last_hidden[0], last_hidden[1]), dim= 1)
        logits= self.fc(cat_hidden)




        #logits = self.label_classifier(concat_h)
        outputs = (logits,) + outputs[2:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs

# config = AutoConfig.from_pretrained('monologg/kobigbird-bert-base')
# model = R_BigBird(config, 0.1)
# print(model)
# print(config)