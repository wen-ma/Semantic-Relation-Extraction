"""
This code is based on the file in PURE repo:
https://github.com/princeton-nlp/PURE/blob/main/relation/models.py
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel


class RelationModel(nn.Module):
    def __init__(self, config, num_rel_labels):
        super().__init__()
        self.num_labels = num_rel_labels
        self.bert = AutoModel.from_pretrained(config["model_name_or_path"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.layer_norm = torch.nn.LayerNorm(config["hidden_size"] * 2)
        self.classifier = nn.Linear(config["hidden_size"] * 2, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            output_hidden_states=False, output_attentions=False)
        sequence_output = outputs.last_hidden_state
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
