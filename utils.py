import numpy as np

np.random.seed(1234)
import os
from gensim.models import word2vec
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
from sklearn.metrics import f1_score
from time import time
import torch


def train_lstm(
    sequences,
    vocab_sz,
    truncate_len,
    save_path,
    word_embedding_dim=100,
    hidden_dim=100,
    embedding_matrix=None,
):
    if embedding_matrix is not None:
        trim_embedding = np.zeros((vocab_sz + 1, word_embedding_dim))
        trim_embedding[:-1, :] = embedding_matrix[:vocab_sz, :]
        trim_embedding[-1, :] = np.average(embedding_matrix[vocab_sz:, :], axis=0)
    else:
        trim_embedding = None
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    from models_LSTM import LSTMLanguageModel

    model_name = save_path + "/model-final.h5"
    model = LSTMLanguageModel(
        truncate_len - 1, word_embedding_dim, vocab_sz + 1, hidden_dim, trim_embedding
    )
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    if os.path.exists(model_name):
        print(f"Loading model {model_name}...")
        model.load_weights(model_name)
        return model
    x, y = sequences[:, :-1], sequences[:, -1]
    checkpointer = ModelCheckpoint(
        filepath=save_path + "/model-{epoch:02d}.h5", save_weights_only=True, period=1
    )
    model.fit(x, y, batch_size=256, epochs=25, verbose=1, callbacks=[checkpointer])
    model.save_weights(model_name)
    return model


def train_word2vec(
    sentence_matrix,
    vocabulary_inv,
    dataset_name,
    suffix="",
    mode="skipgram",
    num_features=100,
    min_word_count=5,
    context=5,
    embedding_train=None,
):
    model_dir = "./" + dataset_name
    model_name = "embedding_" + suffix + ".p"
    model_name = os.path.join(model_dir, model_name)
    num_workers = 15  # Number of threads to run in parallel
    downsampling = 1e-3
    print("Training Word2Vec model...")

    sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
    if mode == "skipgram":
        sg = 1
        print("Model: skip-gram")
    elif mode == "cbow":
        sg = 0
        print("Model: CBOW")
    embedding_model = word2vec.Word2Vec(
        sentences,
        workers=num_workers,
        sg=sg,
        size=num_features,
        min_count=min_word_count,
        window=context,
        sample=downsampling,
    )

    embedding_model.init_sims(replace=True)

    embedding_weights = {
        key: embedding_model[word]
        if word in embedding_model
        else np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
        for key, word in vocabulary_inv.items()
    }
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(f"Saving Word2Vec weights to {model_name}")
    pickle.dump(embedding_weights, open(model_name, "wb"))
    return embedding_weights


def train_class_embedding(
    x,
    vocabulary_inv,
    dataset_name,
    node,
    suffix="",
    mode="skipgram",
    num_features=100,
    min_word_count=5,
    context=5,
):
    print(f"Training embedding for node {node.name}")

    model_dir = "./" + dataset_name
    model_name = "embedding_" + node.name + suffix + ".p"
    model_name = os.path.join(model_dir, model_name)
    if os.path.exists(model_name):
        print(f"Loading existing Word2Vec embedding {model_name}...")
        embedding_weights = pickle.load(open(model_name, "rb"))
        assert len(vocabulary_inv) == len(
            embedding_weights
        ), f"Old word embedding model! Please delete {model_name} and re-run!"
    else:
        suffix = node.name + suffix
        embedding_weights = train_word2vec(
            x,
            vocabulary_inv,
            dataset_name,
            suffix,
            mode,
            num_features,
            min_word_count,
            context,
        )
    embedding_mat = np.array(
        [np.array(embedding_weights[word]) for word in vocabulary_inv]
    )
    node.embedding = embedding_mat


def proceed_level(
    wstc,
    args,
    pretrain_epochs,
    class_tree,
    level,
    need_step2,
    use_multihead,
):
    print(f"\n### Proceeding level {level} ###")
    dataset = args.dataset
    sup_source = args.sup_source
    batch_size = args.batch_size
    parents = class_tree.find_at_level(level)
    parents_names = [parent.name for parent in parents]
    print(f"Nodes: {parents_names}")

    for parent in parents:
        # initialize classifiers in hierarchy
        print("\n### Input preparation ###")
        wstc.instantiate(class_tree=parent, use_class_embedding=use_multihead)

        save_dir = f"./results/{dataset}/{sup_source}/level_{level}"

        if parent.model is not None:
            # load pseudo_docs with labels
            seed_docs_labels = []
            with open(
                os.path.join(save_dir, f"{parent.name}_pseudo_docs_labels.pkl"),
                "rb",
            ) as f:
                seed_docs_labels = pickle.load(f)

            perm_seed_docs_labels = np.random.permutation(seed_docs_labels)
            seed_docs, seed_label = zip(*perm_seed_docs_labels)
            seed_docs = list(seed_docs)
            seed_label = list(seed_label)

            # TODO: Remove hard code
            pretrain_epochs = 3
            batch_size = 16

            print("\n### Phase 1: pre-training with pseudo documents ###")
            print(f"Pretraining node {parent.name}")

            wstc.pretrain(
                x=seed_docs,
                pretrain_labels=seed_label,
                model=parent.model,
                epochs=pretrain_epochs,
                batch_size=batch_size,
            )

            if need_step2:
                print("\n### Phase 2: self-training ###")
                distilled_model = wstc.distill(
                    x=seed_docs,
                    pretrain_labels=seed_label,
                    model=parent.model,
                    epochs=pretrain_epochs,
                    batch_size=batch_size,
                )
                parent.model = distilled_model

            # save the model
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(
                    parent.model.state_dict(), f"{save_dir}/bert_{parent.name}.pt"
                )


def probabilities_dot_product(all_logits, batch_size):
    # dot product
    s = [all_logits[0][0].softmax(1)]
    for lv in range(len(all_logits) - 1):
        child_num_node = len(all_logits[lv + 1])
        tmp_s = [[None for _ in range(batch_size)] for _ in range(child_num_node)]
        for b in range(batch_size):
            for idx in range(child_num_node):
                product = torch.mul(
                    s[lv][b][idx], all_logits[lv + 1][idx][b].softmax(0)
                )
                tmp_s[idx][b] = product
        for idx in range(child_num_node):
            tmp_s[idx] = torch.stack(tmp_s[idx], 0)
        s.append(torch.cat(tmp_s, 1))
    return s


def f1(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    return f1_macro, f1_micro


def write_output(y_pred, perm, class_tree, write_path):
    invperm = np.zeros(len(perm), dtype="int32")
    for i, v in enumerate(perm):
        invperm[v] = i
    y_pred = np.array(y_pred)[invperm]
    label2name = {}
    for i in range(class_tree.get_size() - 1):
        label2name[i] = class_tree.find(i).name

    def get_all_labels(leaf_label):
        ancestors = class_tree.find(leaf_label).find_ancestors()
        ancestors.reverse()
        ancestors_names = [a.name for a in ancestors]
        ancestors_names.append(label2name[leaf_label])
        return ancestors_names

    with open(os.path.join(write_path, "out.txt"), "w") as f:
        for label in y_pred:
            labels = get_all_labels(label)
            if len(labels) > 0:
                out_str = "\t".join(labels)
            else:
                out_str = class_tree.name
            f.write(out_str + "\n")
    print(
        "Classification results are written in {}".format(
            os.path.join(write_path, "out.txt")
        )
    )
