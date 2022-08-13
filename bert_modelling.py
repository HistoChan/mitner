from transformers import BertForSequenceClassification
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, Softmax
import numpy as np


class CustomBERTClassifier(BertForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        loss = None
        if labels is not None:
            if labels.shape == logits.shape:
                predict = Softmax(dim=1)(logits)
                loss_fct = KLDivLoss(reduction="batchmean")
                loss = loss_fct(predict.log(), labels)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)