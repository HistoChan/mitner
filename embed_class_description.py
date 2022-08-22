from transformers import BertTokenizer, BertModel
import torch
import pickle
import csv
from load_data import preprocess_doc, create_tensors

truncate_doc_len = 512
MODEL_TYPE = "bert-base-uncased"


def embed_class_description(dataset="nyt", def_source="wiki"):
    description_dir = f"./{dataset}/label_def.csv"
    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

    cls_def = {}

    with open(description_dir, "r", encoding="utf-8") as description_file:
        reader = csv.reader(description_file)
        next(reader)  # header
        for row in reader:
            # use wiki or dictionary
            definition = row[1] if def_source == "wiki" else row[2]
            data = preprocess_doc(definition)
            data = definition.split(" ")
            trun_data = data[: truncate_doc_len - 2]
            cls_def[row[0]] = trun_data
            # print(trun_data)
    input_ids, attention_masks = create_tensors(
        cls_def.values(), tokenizer, truncate_doc_len
    )

    model = BertModel.from_pretrained(
        MODEL_TYPE,
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_masks)
        embedding = outputs[1]

    cls_embedding = list(zip(cls_def.keys(), embedding))
    output_dir = f"./{dataset}/label_embedding.pkl"
    with open(output_dir, "wb") as outp:
        pickle.dump(cls_embedding, outp, pickle.HIGHEST_PROTOCOL)