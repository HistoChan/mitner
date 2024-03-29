# To suppress Future warning
import warnings
import os
import numpy as np

np.random.seed(1234)
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# from models import WSTC

from models import WSTC
from utils import proceed_level, write_output
from transformers import BertTokenizer


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description="main", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ### Basic settings ###
    # dataset selection: New York Times (default), arXiv and Yelp Review
    parser.add_argument("--dataset", default="nyt", choices=["nyt", "arxiv", "yelp"])
    parser.add_argument("--need_step2", default="n", choices=["y", "n"])
    parser.add_argument("--multihead", default="n", choices=["y", "n"])
    args = parser.parse_args()
    dataset = args.dataset
    need_step2 = args.need_step2 == "y"
    use_multihead = args.multihead == "y"

    # load preprocessed data
    import_dirs = [
        "./" + dataset + d + ".pkl"
        for d in ["_arg", "_x_input_ids", "_x_attention_masks", "_y"]
    ]
    with open(import_dirs[0], "rb") as outp:
        args_dict = pickle.load(outp)
        args = args_dict["args"]
        class_tree = args_dict["class_tree"]
        max_level = args_dict["max_level"]
        perm = args_dict["perm"]
        pretrain_epochs = args_dict["pretrain_epochs"]
        vocab_sz = args_dict["vocab_sz"]
        word_embedding_dim = args_dict["word_embedding_dim"]
    with open(import_dirs[1], "rb") as outp:
        x_input_ids = pickle.load(outp)
    with open(import_dirs[2], "rb") as outp:
        x_attention_masks = pickle.load(outp)
    with open(import_dirs[3], "rb") as outp:
        y = pickle.load(outp)

    # get pre-train tokenizer
    MODEL_TYPE = "bert-base-uncased"
    SEQ_LEN = 256
    BATCH_SIZE = 16
    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

    wstc = WSTC(
        class_tree=class_tree,
        max_level=max_level,
        sup_source=args.sup_source,
        y=None,
        block_thre=args.gamma,
        block_level=args.block_level,
        tokenizer=tokenizer,
    )

    for level in range(max_level):
        proceed_level(
            wstc=wstc,
            args=args,
            pretrain_epochs=pretrain_epochs,
            class_tree=class_tree,
            level=level,
            need_step2=need_step2,
            use_multihead=use_multihead,
        )

    y_pred = wstc.ensemble(args, x_input_ids, x_attention_masks, max_level)

    write_output(y_pred, perm, class_tree, "./" + dataset)
