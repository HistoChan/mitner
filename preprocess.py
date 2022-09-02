import numpy as np

np.random.seed(1234)
import pickle
import argparse
import os
from transformers import BertTokenizer
from load_data import (
    load_dataset,
    convert_LSTM_token_to_text,
    load_data_BERT,
    add_embedding_class_description,
)
from utils import train_class_embedding, train_lstm
from gen import augment, bow_pseudodocs, lstm_pseudodocs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="main", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ### Basic settings ###

    # dataset selection: New York Times (default), arXiv and Yelp Review
    parser.add_argument("--dataset", default="nyt", choices=["nyt", "arxiv", "yelp"])
    # weak supervision selection: class-related keywords (default) and labeled documents
    parser.add_argument(
        "--sup_source", default="keywords", choices=["keywords", "docs"]
    )
    # class description source
    parser.add_argument("--def_source", default="wiki", choices=["wiki", "dict"])
    # the class tree level to proceed until: None (default) = maximum possible level
    parser.add_argument("--max_level", default=None, type=int)
    # the highest class tree level that documents can be assigned to: 1 (default)
    parser.add_argument("--block_level", default=1, type=int)
    # whether ground truth labels are available for evaluation: All (default, all documents have ground truth for evaluation),
    # Subset (a subset of documents have ground truth for evaluation) and None (no ground truth)
    parser.add_argument("--with_eval", default="All", choices=["All", "Subset", "None"])

    ### Training settings ###

    # mini-batch size for both pre-training and self-training: 256 (default)
    parser.add_argument("--batch_size", default=256, type=int)
    # maximum self-training iterations: 5000 (default)
    parser.add_argument("--maxiter", default="50000,50000")
    # pre-training epochs: None (default)
    parser.add_argument("--pretrain_epochs", default=None, type=int)
    # self-training update interval: None (default)
    parser.add_argument("--update_interval", default=None, type=int)
    # pseudo document generation method: bow (Bag-of-words, default) or lstm (LSTM language model)
    parser.add_argument("--pseudo", default="bow", choices=["bow", "lstm"])

    ### Hyperparameters settings ###

    # background word distribution weight (alpha): 0.2 (default)
    parser.add_argument("--alpha", default=0.2, type=float)
    # number of generated pseudo documents per class (beta): 500 (default)
    parser.add_argument("--beta", default=500, type=int)
    # self-training stopping criterion (delta): 0.1 (default)
    parser.add_argument("--delta", default=0.1, type=float)
    # normalized entropy threshold for blocking: 1.0 (default)
    parser.add_argument("--gamma", default=1.0, type=float)

    args = parser.parse_args()
    print(args)

    alpha = args.alpha
    beta = args.beta
    delta = args.delta

    word_embedding_dim = 100

    if args.dataset == "nyt":
        update_interval = 30
        pretrain_epochs = 30
        self_lr = 5e-4
        max_doc_length = 1500
        max_sent_length = 40
        common_words = 10000
    if args.dataset == "arxiv":
        update_interval = 30
        pretrain_epochs = 30
        self_lr = 5e-4
        max_doc_length = 300
        max_sent_length = 50
        common_words = 10000
    if args.dataset == "yelp":
        update_interval = 60
        pretrain_epochs = 30
        self_lr = 5e-3
        max_doc_length = 500
        max_sent_length = 35
        common_words = 5000
    decay = 1e-6

    if args.sup_source == "docs":
        expand_num = 0
    else:
        expand_num = None
    if args.update_interval is not None:
        update_interval = args.update_interval
    if args.pretrain_epochs is not None:
        pretrain_epochs = args.pretrain_epochs

    # dataset loading: returning class tree and data related info
    (
        x,
        y,
        sequences,
        class_tree,
        word_counts,
        vocabulary,
        vocabulary_inv_list,
        len_avg,
        len_std,
        perm,
    ) = load_dataset(
        args.dataset,
        sup_source=args.sup_source,
        common_words=common_words,
        truncate_doc_len=max_doc_length,
        truncate_sent_len=max_sent_length,
        with_eval=args.with_eval,
    )
    print(f"Checkpoint 1: Class Tree finished loading.")

    assert max_doc_length > len_avg, f"max_doc_length should be greater than {len_avg}"

    np.random.seed(1234)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    vocab_sz = len(vocabulary_inv)

    print(f"x shape: {x.shape}")

    max_level = class_tree.get_height() if args.max_level is None else args.max_level

    # background array for LSTM language model generation
    total_counts = sum(word_counts[ele] for ele in word_counts)
    total_counts -= word_counts[vocabulary_inv_list[0]]
    background_array = np.zeros(vocab_sz)
    for i in range(1, vocab_sz):
        background_array[i] = word_counts[vocabulary_inv[i]] / total_counts
    print(f"Checkpoint 2: Background array generated")

    # pseudo-document generation:
    print("\n### Input preparation ###")
    for level in range(max_level):
        print(f"\n### Proceeding level {level} ###")
        dataset = args.dataset
        sup_source = args.sup_source
        parents = class_tree.find_at_level(level)
        parents_names = [parent.name for parent in parents]
        print(f"Nodes: {parents_names}")

        for parent in parents:
            # initialize classifiers in hierarchy

            if class_tree.embedding is None:
                train_class_embedding(
                    x, vocabulary_inv, dataset_name=args.dataset, node=class_tree
                )
            parent.embedding = class_tree.embedding

            save_dir = f"./results/{dataset}/{sup_source}/level_{level}"

            if len(parent.children) > 1:

                print(
                    "\n### Phase 0: vMF distribution fitting & pseudo document generation ###"
                )

                if args.pseudo == "bow":
                    print("Pseudo documents generation (Method: Bag-of-words)...")
                    seed_docs, seed_label = bow_pseudodocs(
                        parent.children,
                        expand_num,
                        background_array,
                        max_doc_length,
                        len_avg,
                        len_std,
                        beta,  # num_doc
                        alpha,  # interp_weight
                        vocabulary_inv,
                        parent.embedding,
                        save_dir,
                    )
                elif args.pseudo == "lstm":
                    print(
                        "Pseudo documents generation (Method: LSTM language model)..."
                    )
                    lm = train_lstm(
                        sequences,
                        common_words,
                        max_sent_length,
                        f"./{dataset}/lm",
                        embedding_matrix=class_tree.embedding,
                    )
                    seed_docs, seed_label = lstm_pseudodocs(
                        parent,
                        expand_num,
                        max_doc_length,
                        len_avg,
                        max_sent_length,
                        len_std,
                        beta,  # num_doc
                        alpha,  # interp_weight
                        vocabulary_inv,
                        lm,
                        common_words,
                        save_dir,
                    )

                num_real_doc = len(seed_docs) / 5

                if sup_source == "docs":
                    real_seed_docs, real_seed_label = augment(
                        x, parent.children, num_real_doc
                    )
                    print(
                        f"Labeled docs {len(real_seed_docs)} + Pseudo docs {len(seed_docs)}"
                    )
                    seed_docs = np.concatenate((seed_docs, real_seed_docs), axis=0)
                    seed_label = np.concatenate((seed_label, real_seed_label), axis=0)
                with open(
                    os.path.join(save_dir, f"{parent.name}_pseudo_docs_labels.pkl"),
                    "wb",
                ) as f:
                    seed_docs_text = convert_LSTM_token_to_text(
                        seed_docs, vocabulary_inv
                    )
                    doc_labels = list(zip(seed_docs_text, seed_label))
                    pickle.dump(doc_labels, f, protocol=4)

                print("Finished pseudo documents generation.")

    print(f"Checkpoint 3: Finished ALL pseudo documents generation.")

    # x in BERT embedding:
    MODEL_TYPE = "bert-base-uncased"
    tokenizer_basic = BertTokenizer.from_pretrained(MODEL_TYPE)
    _, input_ids, attention_masks = load_data_BERT(args.dataset, tokenizer_basic)
    print(f"Checkpoint 4: BERT embedding finished.")

    # NEW: class description embedding:
    add_embedding_class_description(class_tree, args.dataset, args.def_source)
    print(f"Checkpoint 5: Added embedding class description.")

    # DONE: Output 4 package (no need to re-analyze the dataset in the future)
    export_dirs = ["_arg.pkl", "_x_input_ids.pkl", "_x_attention_masks.pkl", "_y.pkl"]
    args_dict = {
        "args": args,
        "class_tree": class_tree,
        "max_level": max_level,
        "perm": perm,
        "pretrain_epochs": pretrain_epochs,
        "vocab_sz": vocab_sz,
        "word_embedding_dim": word_embedding_dim,
    }
    export_objs = [args_dict, input_ids, attention_masks, y]
    export_zip = list(zip(export_dirs, export_objs))

    def export_data(directory, obj):
        output_name = args.dataset + directory
        with open(output_name, "wb") as outp:
            pickle.dump(obj, outp, protocol=4)

    for directory, obj in export_zip:
        export_data(directory, obj)
    print("Preprocessing is done!")