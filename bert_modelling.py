from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
)
import torch
from torch.nn import (
    Module,
    CrossEntropyLoss,
    KLDivLoss,
    Softmax,
    ReLU,
    MultiheadAttention,
)
import numpy as np


def get_loss(predict, label):
    # Unity type: Float to Double
    predict = predict.double()
    label = label.double()
    print(predict, label)

    if label.shape == predict.shape:
        predict = predict.log()
        loss_fct = KLDivLoss(reduction="batchmean")
    else:
        loss_fct = CrossEntropyLoss()
    loss = loss_fct(predict, label)
    return loss.double()


class CustomBERTClassifier(BertForSequenceClassification):
    def __init__(self, config, class_embedding=None):
        super().__init__(config)
        self.class_embedding = class_embedding
        self.multihead_attn = MultiheadAttention(768, 12)

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

        # TODO: multihead attention
        # query = pooled_output.view(1, 1, 768)
        # key_value = self.class_embedding.view(5, 1, 768)
        # pooled_output = self.multihead_attn(query, key_value, key_value)

        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        loss = None
        if labels is not None:
            if labels.shape == logits.shape:
                logits = Softmax(dim=1)(logits)
            loss = get_loss(logits, labels)
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


# Another version using DistilBERT
class CustomDistilBERTClassifier(DistilBertForSequenceClassification):
    def __init__(self, config, class_embedding=None):
        super().__init__(config)
        self.class_embedding = class_embedding
        self.multihead_attn = MultiheadAttention(768, 12)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
    ):
        print(input_ids.shape)
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
        )

        # Similar as the DistilBertForSequenceClassification
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        # TODO: multihead attention
        # query = pooled_output.view(1, 1, 768)
        # key_value = self.class_embedding.view(5, 1, 768)
        # pooled_output = self.multihead_attn(query, key_value, key_value)

        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        loss = None
        print("INIT LOSS")
        if labels is not None:
            if labels.shape == logits.shape:
                logits = Softmax(dim=1)(logits)
            loss = get_loss(logits, labels)
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


""" These properties are fixed.
    {"hidden_size": 768, 
    "hidden_act": "gelu", 
    "initializer_range": 0.02, 
    "vocab_size": 30522, 
    "hidden_dropout_prob": 0.1, 
    "num_attention_heads": 12, 
    "type_vocab_size": 2, 
    "max_position_embeddings": 512, 
    "intermediate_size": 3072, 
    "attention_probs_dropout_prob": 0.1}
"""


def get_bert_based(num_hidden_layers=12, num_labels=2, type="Bert"):
    if type == "Bert":
        model = CustomBERTClassifier.from_pretrained(
            "bert-base-uncased",
            num_hidden_layers=num_hidden_layers,
            num_labels=num_labels,  # The number of output labels
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
    elif type == "DistilBert":
        model = CustomDistilBERTClassifier.from_pretrained(
            "distilbert-base-uncased",
            # num_hidden_layers=num_hidden_layers,
            num_labels=num_labels,  # The number of output labels
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
    else:
        model = None  # ERROR
    return model


# expect the parameters of the original model has 12 hidden layers
def load_bert_parameters(model, param_groups):
    num_hidden_layers = model.config.num_hidden_layers
    if num_hidden_layers < 12:

        def params_in_layer(layer_level):
            return [
                f"bert.encoder.layer.{layer_level}.attention.self.query.weight",
                f"bert.encoder.layer.{layer_level}.attention.self.query.bias",
                f"bert.encoder.layer.{layer_level}.attention.self.key.weight",
                f"bert.encoder.layer.{layer_level}.attention.self.key.bias",
                f"bert.encoder.layer.{layer_level}.attention.self.value.weight",
                f"bert.encoder.layer.{layer_level}.attention.self.value.bias",
                f"bert.encoder.layer.{layer_level}.attention.output.dense.weight",
                f"bert.encoder.layer.{layer_level}.attention.output.dense.bias",
                f"bert.encoder.layer.{layer_level}.attention.output.LayerNorm.weight",
                f"bert.encoder.layer.{layer_level}.attention.output.LayerNorm.bias",
                f"bert.encoder.layer.{layer_level}.intermediate.dense.weight",
                f"bert.encoder.layer.{layer_level}.intermediate.dense.bias",
                f"bert.encoder.layer.{layer_level}.output.dense.weight",
                f"bert.encoder.layer.{layer_level}.output.dense.bias",
                f"bert.encoder.layer.{layer_level}.output.LayerNorm.weight",
                f"bert.encoder.layer.{layer_level}.output.LayerNorm.bias",
            ]

        # select layers to load
        from math import floor

        ratio = 12 / num_hidden_layers
        keep_layers = [floor(idx * ratio) for idx in range(num_hidden_layers)]

        counter = 0
        for idx in range(12):
            if idx in keep_layers:
                # keep the layer
                # but change the index in the parameters
                if counter < idx:
                    params_to_remove = params_in_layer(idx)
                    params_to_add = params_in_layer(counter)
                    for key_old, key_new in zip(params_to_remove, params_to_add):
                        param_groups[key_new] = param_groups.pop(key_old)
                counter += 1
            # remove the layer
            else:
                params_to_remove = params_in_layer(idx)
                for key in params_to_remove:
                    param_groups.pop(key)

    model.load_state_dict(param_groups)
    return model


class Distillator(Module):
    def __init__(self, teacher, temperature=1.0):
        super().__init__()
        self.teacher = teacher  # in BERT
        self.student = get_bert_based(num_labels=teacher.config.num_labels, type="Bert")
        # self.student = load_bert_parameters(self.student, teacher.state_dict())
        self.temperature = temperature

    def distillation_loss(self, teacher_logits, student_logits):
        teacher_logits_temp = (teacher_logits / self.temperature).softmax(1)
        student_logits_temp = (student_logits / self.temperature).softmax(1)
        return get_loss(student_logits_temp, teacher_logits_temp)
