import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import re
from itertools import chain
from transformers import BertTokenizer

# get pre-train tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # 英文pretrain(不區分大小寫)
vocab = tokenizer.vocab
print("dict size", len(vocab))


from torch.utils.data import Dataset, random_split

TAG_RE = re.compile(r"<[^>]+>")


def preprocess_text(sen):
    return sen


def readIMDB(path, seg):
    data = []
    return data


label_map = {0: "neg", 1: "pos"}

# create Dataset
class MyDataset(Dataset):
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]
        self.mode = mode
        self.df = readIMDB(
            "aclImdb", mode
        )  # its list [['text1',label],['text2',label],...]
        self.len = len(self.df)
        self.maxlen = 300  # 限制文章長度(若你記憶體夠多也可以不限)
        self.tokenizer = tokenizer  # we will use BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        origin_text = self.df[idx][0]
        text_a = self.df[idx][0]
        text_b = None  # for natural language inference
        label_id = self.df[idx][1]
        label_tensor = torch.tensor(label_id)

        # 建立第一個句子的 BERT tokens
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a[: self.maxlen] + ["[SEP]"]
        len_a = len(word_pieces)

        if text_b is not None:
            tokens_b = self.tokenizer.tokenize(text_b)
            word_pieces += tokens_b + ["[SEP]"]
            len_b = len(word_pieces) - len_a

        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        if text_b is None:
            segments_tensor = torch.tensor([1] * len_a, dtype=torch.long)
        elif text_b is not None:
            segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor, origin_text)

    def __len__(self):
        return self.len


# initialize Dataset
trainset = MyDataset("train", tokenizer=tokenizer)
testset = MyDataset("test", tokenizer=tokenizer)


# split val from trainset
val_size = int(trainset.__len__() * 0.04)  # 比對LSTM 切出1000筆當validation
trainset, valset = random_split(trainset, [trainset.__len__() - val_size, val_size])
print("trainset size:", trainset.__len__())
print("valset size:", valset.__len__())
print("testset size: ", testset.__len__())


# 隨便選一個樣本
sample_idx = 10

# 利用剛剛建立的 Dataset 取出轉換後的 id tensors
tokens_tensor, segments_tensor, label_tensor, origin_text = trainset[sample_idx]

# 將 tokens_tensor 還原成文本
tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())

print("token:\n", tokens, "\n")
print("origin_text:\n", origin_text, "\n")
print("label:", label_map[int(label_tensor.numpy())], "\n")
print("tokens_tensor:\n", tokens_tensor, "\n")
print("segment tensor:\n", segments_tensor)
