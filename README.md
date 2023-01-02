# MITNER

This source code is based on the one used for Weakly-Supervised Hierarchical Text Classification, published in AAAI 2019.

## Requirements

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

The spherecluster library is already downloaded. Please notice that this library does not support scikit-learn version 0.23 or any newer versions.

Also, be sure to download `'punkt'` in python:

```
import nltk
nltk.download('punkt')
```

## Quick Start

```
python main.py --dataset ${dataset} --sup_source ${sup_source} --with_eval ${with_eval} --pseudo ${pseudo}
```

where you need to specify the dataset in `${dataset}`, the weak supervision type in `${sup_source}` (could be one of `['keywords', 'docs']`), the evaluation type in `${with_eval}` and the pseudo document generation method in `${pseudo}`(`bow` uses bag-of-words method introduced in the CIKM paper. `lstm` uses LSTM language model method introduced in the AAAI paper; it generates better-quality pseudo documents, but requires much longer time for training an LSTM language model).

An example run is provided in `test.sh`, which can be executed by

```
./test.sh
```

More advanced settings on training and hyperparameters are commented in `main.py`.

## Implementation Highlights

The concept of the classifier is similar to the original one. The programme first pretrain the local classifier before ensembling them into a global classifier and conducting fine-tuning. This extended version has conducted 3 major changes to the original one:
This extended version has conducted 3 major changes to the original one:

- For all the classifiers, the LSTM models are replaced by BERT-related models, including Bert and DistilBert. Since there are pretrained BERT-like models, it saves the time to train the model to understand the semantics. The final outcome would be using DistilBert instead of Bert as the classifiers because of the consideration of size and efficiency.
- Instead of using the linear classifier in the BertForSequenceClassification (or DistilBertForSequenceClassification), the multi-headed attention mechanism from the Entity Class Description Attention in SpanNER is used.
- Knowledge distillation is used to reduce the size of the classifier. The idea is similar to the teacher-student framework in BOND.

### Pseudo Document Generation

This part is equivalent to the one in the original WeSHClass.

### Local Classifier Pre-Training

A neural classifier Mp for each class Cp if Cp has two or more children classes. In this work the classifier is the variation of the BertForSequenceClassification. Typically, BertForSequenceClassification contains a 12-layer Bert model, followed by a dropout layer and a linear classifier. In this project, the linear classifier is replaced by the multi-headed attention (MHA) mechanism. Using the Entity Class Description Attention in SpanNER, the MHA accepts vectors from the Bert layer as the Query, and the vectors of the class descriptions generated by another Bert model as both the Key and Value. The output labels are obtained by taking the softmax function of the outcome logits.

Although this step is called “pre-training”, there are already available pre-trained Bert models. Thus, in this step we are actually having the first half of the fine-tuning process of the Bert model. Generated pseudo documents with pseudo labels are used just like the original WeSHClass, with minimising the KL divergence from labels of Mp to the pseudo labels.

### Knowledge Distillation on the Local Classifier

This step is optional, but by having knowledge distillation, the final model would be smaller in size with a faster self-training process in the later step. The reason is we replaced the Bert model with the DistilBert model.

At the beginning of the knowledge distillation, we have a pair of teacher-student models - the teacher model Mp[t] is derived from the previous step, and the student model Mp[s] is the variation of the DistilBertForSequenceClassification, similar to the Bert version. The DistilBert is in the pre-trained version as a way of “re-initialising” the model (as suggested by BOND). During the distillation process, both models receive pseudo documents and generate labels. The student model is trained by calculation of the following 3 losses:

- KL divergence loss between labels from the student model and the pseudo labels of the pseudo documents
- Distillation loss: KL divergence loss between labels from the student model and the teacher model
- Cosine embedding loss between the logits from the student model and the teacher model
  Then we feed back the mean loss to the student model.

### Global Classifier: Knowledge Distillation as Self-Training

Knowledge distillation is also used to avoid propagation of misclassifications at higher levels to lower levels. Conditional probability formula is used to get the final prediction.

In this step, we generate a pair of teacher-student models per each local classifier. The distillation process is similar to above step, except:
The real data is used as the input.

We do not input the labels of the real data for training, so we only focus on two losses and take the mean of them (which means, there is no KL divergence loss between labels from the student model and the pseudo labels of the pseudo documents).
After we get all loss in all pairs of teacher-student models, we take the mean of losses of the nodes within the same layer of the class tree, and feed the mean loss to those student models of the nodes.

### Dataset

NYT is mainly used in this study. To implement the multi-headed attention mechanism, the class descriptions of all classes are collected from Wikipedia with the following steps: For a class with label x:
If there is an article with the title exactly as x, get the first paragraph of the article.
Search for similar articles with the topic similar to x. If there is one, get the first paragraph of the article.
If x can be divided as several parts x1,x2,...,xk(for example “stocks and bonds” can be divided as two items - “stock”, “bonds” (“and” is omitted as it carries no meaning in the label)), search for all parts and concatenate them.

There is another class description with the definition from online Cambridge Dictionary for the NYT dataset, which the class description formation process is similar to the above one. In addition, the class description for arXiv dataset using Wikipedia and bags of topics mentioned in arXiv are also prepared. These descriptions are currently unused.

## Reference

```
@inproceedings{meng2018weakly,
  title={Weakly-Supervised Neural Text Classification},
  author={Meng, Yu and Shen, Jiaming and Zhang, Chao and Han, Jiawei},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={983--992},
  year={2018},
  organization={ACM}
}

@inproceedings{meng2019weakly,
  title={Weakly-supervised hierarchical text classification},
  author={Meng, Yu and Shen, Jiaming and Zhang, Chao and Han, Jiawei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={6826--6833},
  year={2019}
}

@article{wang2021learning,
  title={Learning from language description: Low-shot named entity recognition via decomposed framework},
  author={Wang, Yaqing and Chu, Haoda and Zhang, Chao and Gao, Jing},
  journal={arXiv preprint arXiv:2109.05357},
  year={2021}
}

@inproceedings{liang2020bond,
  title={Bond: Bert-assisted open-domain named entity recognition with distant supervision},
  author={Liang, Chen and Yu, Yue and Jiang, Haoming and Er, Siawpeng and Wang, Ruijia and Zhao, Tuo and Zhang, Chao},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1054--1064},
  year={2020}
}
```
