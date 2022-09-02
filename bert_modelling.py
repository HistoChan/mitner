from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
)
import torch
from torch import unsqueeze, squeeze, bmm
from torch.nn import (
    Module,
    CrossEntropyLoss,
    KLDivLoss,
    CosineEmbeddingLoss,
    Softmax,
    ReLU,
    MultiheadAttention,
)

# import numpy as np

# Global loss function. Declare once use many times.
kl_loss_func = KLDivLoss(reduction="batchmean")
ce_loss_func = CrossEntropyLoss()
cos_loss_func = CosineEmbeddingLoss(reduction="mean")


def get_loss(predict, label):
    # Unity type: Float to Double
    predict = predict.double()
    label = label.double()

    if label.shape == predict.shape:
        predict = predict.log()
        loss_fct = kl_loss_func
    else:
        loss_fct = ce_loss_func
    loss = loss_fct(predict, label)
    return loss


class CustomBERTClassifier(BertForSequenceClassification):
    def __init__(self, config, class_embedding=None, device="cpu"):
        super().__init__(config)
        self.class_embedding = (
            class_embedding.to(device) if class_embedding is not None else None
        )
        if self.class_embedding is not None:
            self.multihead_attn = MultiheadAttention(768, 12).to(device)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
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

        if self.class_embedding is not None:
            # use multihead attention: from SpanNER - biencoder.py - get_scores
            batch_size = pooled_output.size(0)
            query = pooled_output.unsqueeze(1)  # (batch size, 1, embed_dim)
            scores = []
            for idx in range(self.num_labels):  # per each class
                embedding_candidate = (
                    self.class_embedding[idx, :].unsqueeze(0).repeat(batch_size, 1)
                )
                key = embedding_candidate.unsqueeze(1)  # (batch size, 1, embed_dim)
                embedding_candidate, _ = self.multihead_attn(
                    query, key, key
                )  # drop the weight
                embedding_candidate = embedding_candidate.squeeze().unsqueeze(
                    2
                )  # num_mention_in_batch x embed_size x 1
                score = bmm(query, embedding_candidate)  # num_mention_in_batch x 1 x 1
                score = squeeze(score).unsqueeze(-1)  # (batch size, 1)
                scores.append(score)
            logits = torch.cat(scores, dim=-1)  # (batch size, num_label)
        else:
            # use Linear classifier
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
    def __init__(self, config, class_embedding=None, device="cpu"):
        super().__init__(config)
        self.class_embedding = (
            class_embedding.to(device) if class_embedding is not None else None
        )
        if self.class_embedding is not None:
            self.multihead_attn = MultiheadAttention(768, 12).to(device)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)

        # Similar as the DistilBertForSequenceClassification
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        # multihead attention
        if self.class_embedding is not None:
            # use multihead attention: from SpanNER - biencoder.py - get_scores
            batch_size = pooled_output.size(0)
            query = pooled_output.unsqueeze(1)  # (batch size, 1, embed_dim)
            scores = []
            for idx in range(self.num_labels):  # per each class
                embedding_candidate = (
                    self.class_embedding[idx, :].unsqueeze(0).repeat(batch_size, 1)
                )
                key = embedding_candidate.unsqueeze(1)  # (batch size, 1, embed_dim)
                embedding_candidate, _ = self.multihead_attn(
                    query, key, key
                )  # drop the weight
                embedding_candidate = embedding_candidate.squeeze().unsqueeze(
                    2
                )  # num_mention_in_batch x embed_size x 1
                score = bmm(query, embedding_candidate)  # num_mention_in_batch x 1 x 1
                score = squeeze(score).unsqueeze(-1)  # (batch size, 1)
                scores.append(score)
            logits = torch.cat(scores, dim=-1)  # (batch size, num_label)
        else:
            # use Linear classifier
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


def get_bert_based(
    num_hidden_layers=12,
    num_labels=2,
    model_type="Bert",
    class_embedding=None,
    device="cpu",
):
    if model_type == "Bert":
        model = CustomBERTClassifier.from_pretrained(
            "bert-base-uncased",
            num_hidden_layers=num_hidden_layers,
            num_labels=num_labels,  # The number of output labels
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            class_embedding=class_embedding,
            device=device,
        )
    elif model_type == "DistilBert":
        model = CustomDistilBERTClassifier.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels,  # The number of output labels
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            class_embedding=class_embedding,
            device=device,
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


class Distiller(Module):
    def __init__(self, teacher, temperature=1.0, device="cpu"):
        super().__init__()
        self.device = device
        self.teacher = teacher.to(device)  # in BERT
        self.student = get_bert_based(
            num_labels=teacher.num_labels,
            model_type="DistilBert",
            class_embedding=teacher.class_embedding,
            device=device,
        ).to(device)
        # self.student = load_bert_parameters(self.student, teacher.state_dict())
        self.temperature = temperature

    def update_teacher(self):
        self.teacher = self.student.to(self.device)
        self.student = get_bert_based(
            num_labels=self.teacher.num_labels,
            model_type="DistilBert",
            class_embedding=self.teacher.class_embedding,
            device=self.device,
        ).to(self.device)

    def distillation_loss(self, teacher_logits, student_logits):
        teacher_logits_temp = (teacher_logits / self.temperature).softmax(1)
        student_logits_temp = (student_logits / self.temperature).softmax(1)
        coefficient = self.temperature ** 2
        return coefficient * get_loss(student_logits_temp, teacher_logits_temp)

    def cosine_embedding_loss(self, teacher_logits, student_logits):
        target = student_logits.new(student_logits.size(0)).fill_(1)
        return cos_loss_func(student_logits, teacher_logits, target)

    def get_total_loss(self, teacher_logits, student_logits, train_loss=None):
        loss = self.distillation_loss(teacher_logits, student_logits)
        loss += self.cosine_embedding_loss(teacher_logits, student_logits)
        if train_loss is None:
            return loss / 2
        else:
            loss += train_loss
            return loss / 3