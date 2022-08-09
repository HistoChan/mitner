import numpy as np

np.random.seed(1234)
import pickle
import argparse

from load_data import load_dataset


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

    assert max_doc_length > len_avg, f"max_doc_length should be greater than {len_avg}"

    np.random.seed(1234)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    vocab_sz = len(vocabulary_inv)

    print(f"x shape: {x.shape}")

    if args.max_level is None:
        max_level = class_tree.get_height()
    else:
        max_level = args.max_level

    # DONE: Output the whole package (no need to re-analyze the dataset in the future)
    output_name = args.dataset + ".pkl"
    with open(output_name, "wb") as outp:
        outp_dict = {
            "alpha": alpha,
            "args": args,
            "beta": beta,
            "class_tree": class_tree,
            "common_words": common_words,
            "decay": decay,
            "delta": delta,
            "expand_num": expand_num,
            "len_avg": len_avg,
            "len_std": len_std,
            "max_doc_length": max_doc_length,
            "max_level": max_level,
            "max_sent_length": max_sent_length,
            "perm": perm,
            "pretrain_epochs": pretrain_epochs,
            "self_lr": self_lr,
            "sequences": sequences,
            "update_interval": update_interval,
            "vocab_sz": vocab_sz,
            "vocabulary": vocabulary,
            "vocabulary_inv": vocabulary_inv,
            "vocabulary_inv_list": vocabulary_inv_list,
            "word_counts": word_counts,
            "word_embedding_dim": word_embedding_dim,
            "x": x,
            "y": y,
        }
        pickle.dump(outp_dict, outp, pickle.HIGHEST_PROTOCOL)
    print("Preprocessing is done!")