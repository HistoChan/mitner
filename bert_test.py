import torch
import time
import datetime
import random
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import (
    TensorDataset,
    random_split,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from load_data import read_file

"""
    Modify from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
"""

MODEL_TYPE = "bert-base-uncased"


SEQ_LEN = 256
BATCH_SIZE = 16

data, y, class_tree = read_file(dataset="nyt")


def get_label_map(tree):
    children = tree.children
    label_map = {}
    for node in children:
        label_map[node.name] = node.label
    return label_map


label_map = get_label_map(class_tree)
label_num = len(label_map)

# get pre-train tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)


def create_tensors(data, tokenizer):
    input_ids = []
    attention_masks = []

    for sentence in data:
        encoded_dict = tokenizer.encode_plus(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
            truncation=True,
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict["attention_mask"])
    return encoded_dict, input_ids, attention_masks


encoded_dict, input_ids, attention_masks = create_tensors(data, tokenizer)

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Get children label acc. to the parent node
labels = []
for label in y:
    labels.extend([idx for idx in label_map.values() if label[idx] > 0])

labels = torch.tensor(labels)
# labels.size()

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)
# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


def create_dataloader(dataset, sampler):
    return DataLoader(dataset, sampler=sampler(dataset), batch_size=BATCH_SIZE)


# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = create_dataloader(train_dataset, RandomSampler)
# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = create_dataloader(val_dataset, SequentialSampler)

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    MODEL_TYPE,
    num_labels=label_num,  # The number of output labels--2 for binary classification.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
EPOCHS = 3

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Set the seed value all over the place to make this reproducible.
seed_val = 12

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Measure the total training time for the whole run.
total_t0 = time.time()

# Function to calculate the accuracy of our predictions vs labels
def flat_measurement(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    acc_score = accuracy_score(labels_flat, pred_flat)
    f1_macro = f1_score(labels_flat, pred_flat, average="macro")
    f1_micro = f1_score(labels_flat, pred_flat, average="micro")
    return acc_score, f1_macro, f1_micro


for epoch_i in range(EPOCHS):
    #               Training
    # Perform one full pass over the training set.
    print("\n======== Epoch {:} / {:} ========".format(epoch_i + 1, EPOCHS))

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Check bug mode:
        if step > BATCH_SIZE:
            break
        # Progress update every 40 batches.
        if step % BATCH_SIZE == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print("  Batch {:>5,}  of  {:>5,}.".format(step, len(train_dataloader)))

        # Unpack this training batch from our dataloader.
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        loss = outputs[0]
        logits = outputs[1]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.

    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("\nRunning Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_f1_micro = 0
    total_f1_macro = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # Unpack this training batch from our dataloader.
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            loss = outputs[0]
            logits = outputs[1]

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        acc_score, f1_macro, f1_micro = flat_measurement(logits, label_ids)
        total_eval_accuracy += acc_score
        total_f1_macro += f1_macro
        total_f1_micro += f1_micro

    # Report the final accuracy for this validation run.
    data_len = len(validation_dataloader)
    avg_val_accuracy = total_eval_accuracy / data_len
    avg_val_f1_macro = total_f1_macro / data_len
    avg_val_f1_micro = total_f1_micro / data_len
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    print("  F1 Marco: {0:.2f}".format(avg_val_f1_macro))
    print("  F1 Micro: {0:.2f}".format(avg_val_f1_micro))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / data_len

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time,
        }
    )
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))