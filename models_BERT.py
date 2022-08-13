import numpy as np

np.random.seed(1234)
import os
from time import time

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import csv

# resolve problem that keras fails to work
# import tensorflow as tf
# from tensorflow import keras
import tensorflow.keras.backend as K

# from keras.engine.topology import Layer
# from tensorflow.keras.layers.merge import Concatenate
from tensorflow.keras.layers import Layer, Concatenate
from tensorflow.keras.layers import (
    Dense,
    Input,
    Convolution1D,
    Embedding,
    GlobalMaxPooling1D,
    LSTM,
    Multiply,
    Lambda,
    Activation,
)
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.initializers import RandomUniform
from utils import f1
from scipy.stats import entropy

from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch.optim import Adam
from torch.nn import Dropout, Linear, Softmax, KLDivLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from load_data import load_data_BERT


def LSTMLanguageModel(
    input_shape, word_embedding_dim, vocab_sz, hidden_dim, embedding_matrix
):
    x = Input(shape=(input_shape,), name="input")
    z = Embedding(
        vocab_sz,
        word_embedding_dim,
        input_length=input_shape,
        weights=[embedding_matrix],
        trainable=False,
    )(x)
    z = LSTM(hidden_dim, activation="relu", return_sequences=True)(z)
    z = LSTM(hidden_dim, activation="relu")(z)
    z = Dense(vocab_sz, activation="softmax")(z)
    model = Model(inputs=x, outputs=z)
    model.summary()
    return Model(inputs=x, outputs=z)


def CustomBERTClassifier(num_labels):
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,  # The number of output labels
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )


loss_f = KLDivLoss(reduction="batchmean")


def get_KLDivLoss(logits, labels, num_labels, inclass, misclass):
    predict = Softmax(dim=1)(logits)
    actual = torch.Tensor(
        [
            [inclass if idx == l else misclass for idx in range(num_labels)]
            for l in labels
        ]
    )

    # change the loss function to KLDIVLOSS
    return loss_f(predict.log(), actual)


class WSTC(object):
    def __init__(
        self,
        input_shape,
        class_tree,
        max_level,
        sup_source,
        init=RandomUniform(minval=-0.01, maxval=0.01),
        y=None,
        vocab_sz=None,
        word_embedding_dim=100,
        blocking_perc=0,
        block_thre=1.0,
        block_level=1,
        tokenizer=None,
    ):

        super(WSTC, self).__init__()

        self.input_shape = input_shape
        self.class_tree = class_tree
        self.y = y
        if type(y) == dict:
            self.eval_set = np.array([ele for ele in y])
        else:
            self.eval_set = None
        self.vocab_sz = vocab_sz
        self.block_level = block_level
        self.block_thre = block_thre
        self.block_label = {}
        self.siblings_map = {}
        self.x = Input(shape=(input_shape[1],), name="input")
        self.model = []
        self.sup_dict = {}
        if sup_source == "docs":
            n_classes = class_tree.get_size() - 1
            leaves = class_tree.find_leaves()
            for leaf in leaves:
                current = np.zeros(n_classes)
                for i in class_tree.name2label(leaf.name):
                    current[i] = 1.0
                for idx in leaf.sup_idx:
                    self.sup_dict[idx] = current
        self.tokenizer = tokenizer

    def instantiate(
        self,
        class_tree,
        filter_sizes=[2, 3, 4, 5],
        num_filters=20,
        word_trainable=False,
        word_embedding_dim=100,
        hidden_dim=20,
        act="relu",
        init=RandomUniform(minval=-0.01, maxval=0.01),
    ):
        num_children = len(class_tree.children)
        if num_children <= 1:
            class_tree.model = None
        else:
            # TODO: BERT
            # Load BertForSequenceClassification, the pretrained BERT model with a single
            # linear classification layer on top.
            class_tree.model = CustomBERTClassifier(num_children)

    def ensemble(self, class_tree, level, input_shape, parent_output):
        print(f"ENTER ensemble with level {level}")
        print(f"ENTER ensemble with class_tree {class_tree.name}")
        print(f"parent_output {parent_output}")
        outputs = []
        if class_tree.model:
            print("class_tree.model: Ensemble part I")
            # TODO: here
            y_curr = parent_output  # class_tree.model(self.x)
            if parent_output is not None:
                y_curr = Multiply()([parent_output, y_curr])
        else:
            y_curr = parent_output

        if level == 0:
            outputs.append(y_curr)
        else:
            print("level !== 0 Ensemble part II")
            for i, child in enumerate(class_tree.children):
                outputs += self.ensemble(
                    child, level - 1, input_shape, None  # IndexLayer(i)(y_curr)
                )
        return outputs

    # TODO: Check if change?
    def ensemble_classifier(self, level):
        outputs = self.ensemble(self.class_tree, level, self.input_shape[1], None)
        print(f"outputs {outputs}")
        outputs = [
            ExpanLayer(-1)(output) if len(output.get_shape()) < 2 else output
            for output in outputs
        ]
        print(f"outputs {outputs}")
        z = Concatenate()(outputs) if len(outputs) > 1 else outputs[0]
        return Model(inputs=self.x, outputs=z)

    # TODO: LSTM to BERTensemble_classifier
    # Since the BERT Classifier is already pre-trained, this part would be
    # fine-tuning the BERT Classifier
    def pretrain(
        self,
        x,
        pretrain_labels,
        model,
        optimizer="adam",  # not use this one
        loss="kld",  # not use this one
        epochs=200,
        batch_size=256,
        save_dir=None,
        suffix="",
    ):
        optimizer = Adam(model.parameters(), lr=1e-5)
        epochs = 1  # 3 TODO: not hard code
        batch_size = 1  # 16 TODO: not hard code

        inclass = max(pretrain_labels[0][:2])
        misclass = min(pretrain_labels[0][:2])

        # input tensors
        tokenizer = self.tokenizer
        tokenizer, input_ids, attention_masks = load_data_BERT(x, tokenizer)
        # output tensors
        labels = torch.tensor(np.argmax(pretrain_labels, axis=1).flatten())
        # Pack up as a dataset
        dataset = TensorDataset(input_ids, attention_masks, labels)
        train_dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset), batch_size=batch_size
        )

        # Tell pytorch to run this model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        model = model.to(device)

        t0 = time()
        print("\nPretraining...")
        for epoch_i in range(epochs):
            total_train_loss = 0
            #               Training
            # Perform one full pass over the training set.
            print("\n======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                # TODO: Debug
                if step > batch_size:
                    break
                # Progress update.
                if step % batch_size == 0 and not step == 0:
                    # Report progress.
                    print(
                        "  Batch {:>5,}  of  {:>5,}.".format(
                            step, len(train_dataloader)
                        )
                    )

                batch_input_ids = batch[0].to(device)
                batch_input_mask = batch[1].to(device)
                batch_labels = batch[2].to(device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                model.zero_grad()

                outputs = model(
                    batch_input_ids,
                    token_type_ids=None,
                    attention_mask=batch_input_mask,
                    labels=batch_labels,
                )
                logits = outputs[1]
                # calculate loss manually
                loss = get_KLDivLoss(
                    logits, batch_labels, model.num_labels, inclass, misclass
                )
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print(f"Pretraining time: {time() - t0:.2f}s")
            # model.fit(x, pretrain_labels, batch_size=batch_size, epochs=epochs)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), f"{save_dir}/pretrained_bert_{suffix}.pt")

    def load_weights(self, weights, level):
        print(f"Loading weights @ level {level}")
        self.model[level].load_weights(weights)

    def load_pretrain(self, weights, model):
        model.load_weights(weights)

    def extract_label(self, y, level):
        if type(level) is int:
            relevant_nodes = self.class_tree.find_at_level(level)
            relevant_labels = [relevant_node.label for relevant_node in relevant_nodes]
        else:
            relevant_labels = []
            for i in level:
                relevant_nodes = self.class_tree.find_at_level(i)
                relevant_labels += [
                    relevant_node.label for relevant_node in relevant_nodes
                ]
        if type(y) is dict:
            y_ret = {}
            for key in y:
                y_ret[key] = y[key][relevant_labels]
        else:
            y_ret = y[:, relevant_labels]
        return y_ret

    def predict(self, x, level):
        q = self.model[level].predict(x, verbose=0)
        return q.argmax(1)

    def expand_pred(self, q_pred, level, cur_idx):
        y_expanded = np.zeros((self.input_shape[0], q_pred.shape[1]))
        if level not in self.siblings_map:
            self.siblings_map[level] = self.class_tree.siblings_at_level(level)
        siblings_map = self.siblings_map[level]
        block_idx = []
        for i, q in enumerate(q_pred):
            pred = np.argmax(q)
            idx = cur_idx[i]
            if (
                level >= self.block_level
                and self.block_thre < 1.0
                and idx not in self.sup_dict
            ):
                siblings = siblings_map[pred]
                siblings_pred = q[siblings] / np.sum(q[siblings])
                if len(siblings) >= 2:
                    conf_val = entropy(siblings_pred) / np.log(len(siblings))
                else:
                    conf_val = 0
                if conf_val > self.block_thre:
                    block_idx.append(idx)
                else:
                    y_expanded[idx, pred] = 1.0
            else:
                y_expanded[idx, pred] = 1.0
        if self.block_label:
            blocked = [idx for idx in self.block_label]
            blocked_labels = np.array([label for label in self.block_label.values()])
            blocked_labels = self.extract_label(blocked_labels, level + 1)
            y_expanded[blocked, :] = blocked_labels
        return y_expanded, block_idx

    def aggregate_pred(self, q_all, level, block_idx, cur_idx, agg="All"):
        leaves = self.class_tree.find_at_level(level + 1)
        leaves_labels = [leaf.label for leaf in leaves]
        parents = self.class_tree.find_at_level(level)
        parents_labels = [parent.label for parent in parents]
        ancestor_dict = {}
        for leaf in leaves:
            ancestors = leaf.find_ancestors()
            ancestor_dict[leaf.label] = [ancestor.label for ancestor in ancestors]
        for parent in parents:
            ancestors = parent.find_ancestors()
            ancestor_dict[parent.label] = [ancestor.label for ancestor in ancestors]
        y_leaf = np.argmax(q_all[:, leaves_labels], axis=1)
        y_leaf = [leaves_labels[y] for y in y_leaf]
        if level > 0:
            y_parents = np.argmax(q_all[:, parents_labels], axis=1)
            y_parents = [parents_labels[y] for y in y_parents]
        if agg == "Subset" and self.eval_set is not None:
            cur_eval = [ele for ele in self.eval_set if ele in cur_idx]
            inv_cur_idx = {i: idx for idx, i in enumerate(cur_idx)}
            y_aggregate = np.zeros((len(cur_eval), q_all.shape[1]))
            for i, raw_idx in enumerate(cur_eval):
                idx = inv_cur_idx[raw_idx]
                if raw_idx not in block_idx:
                    y_aggregate[i, y_leaf[idx]] = 1.0
                    for ancestor in ancestor_dict[y_leaf[idx]]:
                        y_aggregate[i, ancestor] = 1.0
                else:
                    if level > 0:
                        y_aggregate[i, y_parents[idx]] = 1.0
                        for ancestor in ancestor_dict[y_parents[idx]]:
                            y_aggregate[i, ancestor] = 1.0
        else:
            y_aggregate = np.zeros((self.input_shape[0], q_all.shape[1]))
            for i in range(len(q_all)):
                idx = cur_idx[i]
                if idx not in block_idx:
                    y_aggregate[idx, y_leaf[i]] = 1.0
                    for ancestor in ancestor_dict[y_leaf[i]]:
                        y_aggregate[idx, ancestor] = 1.0
                else:
                    if level > 0:
                        y_aggregate[idx, y_parents[i]] = 1.0
                        for ancestor in ancestor_dict[y_parents[i]]:
                            y_aggregate[idx, ancestor] = 1.0
            if self.block_label:
                blocked = [idx for idx in self.block_label]
                blocked_labels = np.array(
                    [label for label in self.block_label.values()]
                )
                blocked_labels = self.extract_label(blocked_labels, range(1, level + 2))
                y_aggregate[blocked, :] = blocked_labels
        return y_aggregate

    def record_block(self, block_idx, y_pred_agg):
        n_classes = self.class_tree.get_size() - 1
        for idx in block_idx:
            self.block_label[idx] = np.zeros(n_classes)
            self.block_label[idx][: len(y_pred_agg[idx])] = y_pred_agg[idx]

    def target_distribution(self, q, nonblock, sup_level, power=2):
        q = q[nonblock]
        weight = q ** power / q.sum(axis=0)
        p = (weight.T / weight.sum(axis=1)).T
        inv_nonblock = {k: v for v, k in enumerate(nonblock)}
        for i in sup_level:
            mapped_i = inv_nonblock[i]
            p[mapped_i] = sup_level[i]
        return p

    def compile(self, level, optimizer="sgd", loss="kld"):
        self.model[level].compile(optimizer=optimizer, loss=loss)
        # print(f"\nLevel {level} model summary: ")
        # self.model[level].summary()

    def fit(
        self,
        x,
        level,
        maxiter=5e4,
        batch_size=256,
        tol=0.25,  # 0.1
        power=2,
        update_interval=100,
        save_dir=None,
        save_suffix="",
    ):
        print("fitting...")
        """model = self.model[level]
        print(f"Update interval: {update_interval}")

        cur_idx = np.array(
            [idx for idx in range(x.shape[0]) if idx not in self.block_label]
        )
        x = x[cur_idx]
        y = self.y

        # logging files
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfiles = []
        logwriters = []
        for i in range(level + 2):
            if i <= level:
                logfile = open(
                    save_dir + f"/self_training_log_level_{i}{save_suffix}.csv", "w"
                )
            else:
                logfile = open(
                    save_dir + f"/self_training_log_all{save_suffix}.csv", "w"
                )
            logwriter = csv.DictWriter(
                logfile, fieldnames=["iter", "f1_macro", "f1_micro"]
            )
            logwriter.writeheader()
            logfiles.append(logfile)
            logwriters.append(logwriter)

        index = 0

        if y is not None:
            if self.eval_set is not None:
                cur_eval = [idx for idx in self.eval_set if idx in cur_idx]
                y = np.array([y[idx] for idx in cur_eval])
            y_all = []
            label_all = []
            for i in range(level + 1):
                y_curr = self.extract_label(y, i + 1)
                y_all.append(y_curr)
                nodes = self.class_tree.find_at_level(i + 1)
                label_all += [node.label for node in nodes]
            y = y[:, label_all]

        mapped_sup_dict_level = {}
        if len(self.sup_dict) > 0:
            sup_dict_level = self.extract_label(self.sup_dict, level + 1)
            inv_cur_idx = {i: idx for idx, i in enumerate(cur_idx)}
            for key in sup_dict_level:
                mapped_sup_dict_level[inv_cur_idx[key]] = sup_dict_level[key]

        for ite in range(int(maxiter)):
            try:
                if ite % update_interval == 0:
                    print(f"\nIter {ite}: ")
                    y_pred_all = []
                    q_all = np.zeros((len(x), 0))
                    for i in range(level + 1):
                        q_i = self.model[i].predict(x)
                        q_all = np.concatenate((q_all, q_i), axis=1)
                        y_pred_i, block_idx = self.expand_pred(q_i, i, cur_idx)
                        y_pred_all.append(y_pred_i)
                    q = q_i
                    y_pred = y_pred_i
                    if len(block_idx) > 0:
                        print(
                            f"Number of blocked documents back to level {level}: {len(block_idx)}"
                        )
                    y_pred_agg = self.aggregate_pred(q_all, level, block_idx, cur_idx)

                    if y is not None:
                        if self.eval_set is not None:
                            y_pred_agg = self.aggregate_pred(
                                q_all, level, block_idx, cur_idx, agg="Subset"
                            )
                            y_pred_all = [y_pred[cur_eval, :] for y_pred in y_pred_all]
                            for i in range(level + 1):
                                f1_macro, f1_micro = np.round(
                                    f1(y_all[i], y_pred_all[i]), 5
                                )
                                print(
                                    f"Evaluated at subset of size {len(cur_eval)}: f1_macro = {f1_macro}, f1_micro = {f1_micro} @ level {i+1}"
                                )
                                logdict = dict(
                                    iter=ite, f1_macro=f1_macro, f1_micro=f1_micro
                                )
                                logwriters[i].writerow(logdict)
                            f1_macro, f1_micro = np.round(f1(y, y_pred_agg), 5)
                            logdict = dict(
                                iter=ite, f1_macro=f1_macro, f1_micro=f1_micro
                            )
                            logwriters[-1].writerow(logdict)
                            print(
                                f"Evaluated at subset of size {len(cur_eval)}: f1_macro = {f1_macro}, f1_micro = {f1_micro} @ all classes"
                            )
                        else:
                            y_pred_agg = self.aggregate_pred(
                                q_all, level, block_idx, cur_idx
                            )
                            for i in range(level + 1):
                                f1_macro, f1_micro = np.round(
                                    f1(y_all[i], y_pred_all[i]), 5
                                )
                                print(
                                    f"f1_macro = {f1_macro}, f1_micro = {f1_micro} @ level {i+1}"
                                )
                                logdict = dict(
                                    iter=ite, f1_macro=f1_macro, f1_micro=f1_micro
                                )
                                logwriters[i].writerow(logdict)
                            f1_macro, f1_micro = np.round(f1(y, y_pred_agg), 5)
                            logdict = dict(
                                iter=ite, f1_macro=f1_macro, f1_micro=f1_micro
                            )
                            logwriters[-1].writerow(logdict)
                            print(
                                f"f1_macro = {f1_macro}, f1_micro = {f1_micro} @ all classes"
                            )

                    nonblock = np.array(list(set(range(x.shape[0])) - set(block_idx)))
                    x_nonblock = x[nonblock]
                    p_nonblock = self.target_distribution(
                        q, nonblock, mapped_sup_dict_level, power
                    )

                    if ite > 0:
                        change_idx = []
                        for i in range(len(y_pred)):
                            if not np.array_equal(y_pred[i], y_pred_last[i]):
                                change_idx.append(i)
                        y_pred_last = np.copy(y_pred)
                        delta_label = len(change_idx)
                        print(
                            f"Fraction of documents with label changes: {np.round(delta_label/y_pred.shape[0]*100, 3)} %"
                        )

                        if delta_label / y_pred.shape[0] < tol / 100:
                            print(
                                f"\nFraction: {np.round(delta_label / y_pred.shape[0] * 100, 3)} % < tol: {tol} %"
                            )
                            print(
                                "Reached tolerance threshold. Self-training terminated."
                            )
                            break
                    else:
                        y_pred_last = np.copy(y_pred)

                # train on batch
                index_array = np.arange(x_nonblock.shape[0])
                if index * batch_size >= x_nonblock.shape[0]:
                    index = 0
                idx = index_array[
                    index
                    * batch_size : min((index + 1) * batch_size, x_nonblock.shape[0])
                ]
                try:
                    assert len(idx) > 0
                except AssertionError:
                    print(f"Error @ index {index}")
                model.train_on_batch(x=x_nonblock[idx], y=p_nonblock[idx])
                index = (
                    index + 1 if (index + 1) * batch_size < x_nonblock.shape[0] else 0
                )
                ite += 1

            except KeyboardInterrupt:
                print("\nKeyboard interrupt! Self-training terminated.")
                break

        for logfile in logfiles:
            logfile.close()

        if save_dir is not None:
            model.save_weights(save_dir + "/final.h5")
            print(f"Final model saved to: {save_dir}/final.h5")
        q_all = np.zeros((len(x), 0))
        for i in range(level + 1):
            q_i = self.model[i].predict(x)
            q_all = np.concatenate((q_all, q_i), axis=1)
        y_pred_agg = self.aggregate_pred(q_all, level, block_idx, cur_idx)
        self.record_block(block_idx, y_pred_agg)
        return y_pred_agg"""
