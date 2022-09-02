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
import torch
from transformers import BertTokenizer
from utils import write_output


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description="main", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ### Basic settings ###
    # dataset selection: New York Times (default), arXiv and Yelp Review
    parser.add_argument("--dataset", default="nyt", choices=["nyt", "arxiv", "yelp"])
    args = parser.parse_args()
    dataset = args.dataset

    # load preprocessed data
    import_dirs = f"./{dataset}_arg.pkl"
    with open(import_dirs, "rb") as outp:
        args_dict = pickle.load(outp)
        args = args_dict["args"]
        class_tree = args_dict["class_tree"]
        max_level = args_dict["max_level"]
        perm = args_dict["perm"]
        pretrain_epochs = args_dict["pretrain_epochs"]
        vocab_sz = args_dict["vocab_sz"]
        word_embedding_dim = args_dict["word_embedding_dim"]
    import_dirs = f"./{dataset}_x_input_ids.pkl"
    with open(import_dirs[1], "rb") as outp:
        x_input_ids = pickle.load(outp)
    import_dirs = f"./{dataset}_x_attention_masks.pkl"
    with open(import_dirs[2], "rb") as outp:
        x_attention_masks = pickle.load(outp)

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

    # extract proceed_level step 2 here
    for level in range(max_level):
        print(f"\n### Proceeding level {level} ###")
        sup_source = args.sup_source
        batch_size = args.batch_size
        parents = class_tree.find_at_level(level)
        parents_names = [parent.name for parent in parents]
        print(f"Nodes: {parents_names}")

        for parent in parents:
            # initialize classifiers in hierarchy
            print("\n### Input preparation ###")
            wstc.instantiate(class_tree=parent, use_class_embedding=False)
            print("\n### Loading back the pre-trained models")
            parent.model.load_state_dict(torch.load(f"results/bert_{parent.name}.pt"))

    y_pred = wstc.ensemble(args, x_input_ids, x_attention_masks, max_level)

    write_output(y_pred, perm, class_tree, "./" + dataset)