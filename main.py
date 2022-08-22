# To suppress Future warning
import warnings
import os
import numpy as np

np.random.seed(1234)
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# from models import WSTC

from models_BERT import WSTC
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
    # parser.add_argument("--dataset", default="nyt", choices=["nyt", "arxiv", "yelp"])
    # args = parser.parse_args()
    # print(args)

    # output_name = args.dataset + ".pkl"
    output_name = "nyt.pkl"
    outp = open(output_name, "rb")
    outp_dict = pickle.load(outp)

    # get pre-train tokenizer
    MODEL_TYPE = "bert-base-uncased"
    SEQ_LEN = 256
    BATCH_SIZE = 16
    tokenizer_basic = BertTokenizer.from_pretrained(MODEL_TYPE)
    # tokenizer, input_ids, attention_masks = load_dataset_BERT("nyt", tokenizer_basic)

    wstc = WSTC(
        input_shape=outp_dict["x"].shape,
        class_tree=outp_dict["class_tree"],
        max_level=outp_dict["max_level"],
        sup_source=outp_dict["args"].sup_source,
        y=outp_dict["y"],
        vocab_sz=outp_dict["vocab_sz"],
        word_embedding_dim=outp_dict["word_embedding_dim"],
        block_thre=outp_dict["args"].gamma,
        block_level=outp_dict["args"].block_level,
        tokenizer=tokenizer_basic,
    )

    total_counts = sum(
        outp_dict["word_counts"][ele] for ele in outp_dict["word_counts"]
    )
    total_counts -= outp_dict["word_counts"][outp_dict["vocabulary_inv_list"][0]]
    background_array = np.zeros(outp_dict["vocab_sz"])
    for i in range(1, outp_dict["vocab_sz"]):
        background_array[i] = (
            outp_dict["word_counts"][outp_dict["vocabulary_inv"][i]] / total_counts
        )

    for level in range(outp_dict["max_level"]):
        y_pred = proceed_level(
            outp_dict["x"],
            outp_dict["sequences"],
            wstc,
            outp_dict["args"],
            outp_dict["pretrain_epochs"],
            outp_dict["self_lr"],
            outp_dict["decay"],
            outp_dict["update_interval"],
            outp_dict["delta"],
            outp_dict["class_tree"],
            level,
            outp_dict["expand_num"],
            background_array,
            outp_dict["max_doc_length"],
            outp_dict["max_sent_length"],
            outp_dict["len_avg"],
            outp_dict["len_std"],
            outp_dict["beta"],
            outp_dict["alpha"],
            outp_dict["vocabulary_inv"],
            outp_dict["common_words"],
        )
    write_output(
        y_pred,
        outp_dict["perm"],
        outp_dict["class_tree"],
        "./" + outp_dict["args"].dataset,
    )
