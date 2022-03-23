import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BigBirdModel


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


class R_BigBird(BigBirdModel):
    def __init__(self, config, dropout_rate):
        super(R_BigBird, self).__init__(config)
        self.model = AutoModel.from_pretrained('monologg/kobigbird-bert-base')
        self.model_config = AutoConfig.from_pretrained('monologg/kobigbird-bert-base')
        self.model_config.num_labels = 30

        self.cls_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, dropout_rate)
        self.entity_fc_layer1 = FCLayer(self.config.hidden_size, self.config.hidden_size, dropout_rate)
        self.entity_fc_layer2 = FCLayer(self.config.hidden_size, self.config.hidden_size, dropout_rate)

        self.label_classifier = FCLayer(
            self.config.hidden_size * 3,
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

    def forward(self, input_ids, attention_mask, labels, e1_mask, e2_mask):
        outputs = self.model(
            input_ids, attention_mask=attention_mask
        )
        sequence_output = outputs[0]

        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        sentence_representation = self.cls_fc_layer(outputs.pooler_output)

        e1_h = self.entity_fc_layer1(e1_h)
        e2_h = self.entity_fc_layer2(e2_h)

        concat_h = torch.cat([sentence_representation, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        outputs = (loss,) + outputs

        return outputs