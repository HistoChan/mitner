import numpy as np

np.random.seed(1234)
import os
from time import time

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from load_data import BERT_TRUNCATE_LENGTH, create_tensors
from bert_modelling import get_bert_based, Distiller, get_loss, cos_loss_func
from utils import probabilities_dot_product


class WSTC(object):
    def __init__(
        self,
        class_tree,
        max_level,
        sup_source,
        y=None,
        block_thre=1.0,
        block_level=1,
        tokenizer=None,
    ):

        super(WSTC, self).__init__()
        self.class_tree = class_tree
        self.max_level = max_level
        self.y = y
        if type(y) == dict:
            self.eval_set = np.array([ele for ele in y])
        else:
            self.eval_set = None
        self.block_level = block_level
        self.block_thre = block_thre
        self.block_label = {}
        self.siblings_map = {}
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)

    def instantiate(self, class_tree, use_class_embedding=True, model_type="Bert"):
        children = class_tree.children
        num_children = len(children)
        class_embedding = (
            torch.stack([node.embedding_BERT for node in children])
            if use_class_embedding
            else None
        )
        if num_children <= 1:
            class_tree.model = None
        else:
            class_tree.model = get_bert_based(
                num_labels=num_children,
                model_type=model_type,
                class_embedding=class_embedding,
                device=self.device,
            )

    # Since the BERT Classifier is already pre-trained, this part would be
    # fine-tuning the BERT Classifier
    def pretrain(
        self,
        x,
        pretrain_labels,
        model,
        epochs=3,
        batch_size=64,  # ref: https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU
    ):

        optimizer = Adam(model.parameters(), lr=1e-5)

        # input tensors
        input_ids, attention_masks = create_tensors(
            x, self.tokenizer, BERT_TRUNCATE_LENGTH
        )

        # Pack up as a dataset
        labels = torch.tensor(pretrain_labels)
        dataset = TensorDataset(input_ids, attention_masks, labels)
        train_dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset), batch_size=batch_size
        )

        # Tell pytorch to run this model.
        model = model.to(self.device)

        t0 = time()
        print("\nPretraining...")
        for epoch_i in range(epochs):
            total_train_loss = 0
            #               Training
            # Perform one full pass over the training set.
            print("\n======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                # Progress update.
                if step % batch_size == 0 and not step == 0:
                    # Report progress.
                    print(
                        "  Batch {:>5,}  of  {:>5,}.".format(
                            step, len(train_dataloader)
                        )
                    )

                batch_input_ids = batch[0].to(self.device)
                batch_input_mask = batch[1].to(self.device)
                batch_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a backward pass.
                model.zero_grad()

                outputs = model(
                    batch_input_ids,
                    token_type_ids=None,
                    attention_mask=batch_input_mask,
                    labels=batch_labels,
                )
                loss = outputs[0]
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                total_train_loss += loss.item()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                optimizer.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print(f"Pretraining time: {time() - t0:.2f}s")

    # distillation the bert model to DistilBert
    def distill(
        self,
        x,
        pretrain_labels,
        model,
        epochs=3,
        batch_size=16,  # ref: https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU
    ):
        print("Initializing knowledge distillation...")
        distiller = Distiller(model, temperature=2.0, device=self.device)

        optimizer = Adam(distiller.student.parameters(), lr=1e-5)

        # input tensors
        input_ids, attention_masks = create_tensors(
            x, self.tokenizer, BERT_TRUNCATE_LENGTH
        )

        # Pack up as a dataset
        labels = torch.tensor(pretrain_labels)
        dataset = TensorDataset(input_ids, attention_masks, labels)
        train_dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset), batch_size=batch_size
        )

        # Tell pytorch to run this model.
        print("device:", self.device)

        t0 = time()
        print("\nDistillation start...")
        # Tell pytorch to run this model.
        distiller.teacher = distiller.teacher.to(self.device)
        distiller.student = distiller.student.to(self.device)
        for epoch_i in range(epochs):
            total_distill_loss = 0
            #               Training
            print("\n======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                # Progress update.
                if step % batch_size == 0 and not step == 0:
                    # Report progress.
                    print(
                        "  Batch {:>5,}  of  {:>5,}.".format(
                            step, len(train_dataloader)
                        )
                    )

                batch_input_ids = batch[0].to(self.device)
                batch_input_mask = batch[1].to(self.device)
                batch_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a backward pass.
                distiller.student.zero_grad()

                distiller.teacher.eval()
                teacher_outputs = distiller.teacher(
                    batch_input_ids,
                    attention_mask=batch_input_mask,
                )
                teacher_logits = teacher_outputs[0]

                distiller.student.train()
                student_outputs = distiller.student(
                    batch_input_ids,
                    attention_mask=batch_input_mask,
                    labels=batch_labels,
                )
                train_loss = student_outputs[0]
                student_logits = student_outputs[1]

                loss = distiller.get_total_loss(
                    teacher_logits, student_logits, train_loss
                )

                # Perform a backward pass to calculate the gradients.
                loss.backward()
                total_distill_loss += loss.item()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                clip_grad_norm_(distiller.student.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                optimizer.step()

            # Calculate the average loss over all of the batches.
            avg_distill_loss = total_distill_loss / len(train_dataloader)

            # Measure how long this epoch took.
            print("  Average training loss: {0:.2f}".format(avg_distill_loss))
            print(f"Distillation time: {time() - t0:.2f}s")

        # use the student model as the output
        return distiller.student

    def ensemble(self, args, input_ids, attention_masks, level):
        print("\n### Phase 3: ensemble and fine-tune ###")
        save_dir = f"./results/{args.dataset}/{args.sup_source}"
        nodes = [self.class_tree.find_at_level(lv) for lv in range(level)]
        distillers = [
            [Distiller(n.model, device=self.device) for n in lv] for lv in nodes
        ]
        optimizers = [
            [Adam(d.student.parameters(), lr=1e-5) for d in lv] for lv in distillers
        ]
        # TODO: not hard code
        epochs = 3
        batch_size = 16

        # Pack up as a dataset
        dataset = TensorDataset(input_ids, attention_masks)
        train_dataloader = DataLoader(
            dataset, sampler=SequentialSampler(dataset), batch_size=batch_size
        )

        # Tell pytorch to run this model.
        print("device:", self.device)

        t0 = time()
        print("\Self-training start...")

        # Always clear any previously calculated gradients before performing a backward pass.
        def get_teacher_logits(teacher):
            teacher.eval()
            teacher_outputs = teacher(
                batch_input_ids,
                attention_mask=batch_input_mask,
            )
            teacher_logits = teacher_outputs[0]
            return teacher_logits

        def get_student_logits(student):
            student.zero_grad()
            student.train()
            student_outputs = student(
                batch_input_ids,
                attention_mask=batch_input_mask,
            )
            student_logits = student_outputs[0]
            return student_logits

        for epoch_i in range(epochs):
            total_train_loss = 0
            #               Training
            # Perform one full pass over the training set.
            print("\n======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
            for step, batch in enumerate(train_dataloader):
                # Progress update.
                if step % batch_size == 0 and not step == 0:
                    # Report progress.
                    print(
                        "  Batch {:>5,}  of  {:>5,}.".format(
                            step, len(train_dataloader)
                        )
                    )

                batch_input_ids = batch[0].to(self.device)
                batch_input_mask = batch[1].to(self.device)

                all_teacher_logits = [
                    [get_teacher_logits(distiller.teacher) for distiller in level]
                    for level in distillers
                ]
                all_student_logits = [
                    [get_student_logits(distiller.student) for distiller in level]
                    for level in distillers
                ]
                dot_teacher_prob = probabilities_dot_product(
                    all_teacher_logits, batch_size
                )
                dot_student_prob = probabilities_dot_product(
                    all_student_logits, batch_size
                )

                # calculate the loss
                for lv in range(len(dot_teacher_prob)):
                    loss = get_loss(dot_student_prob[lv], dot_teacher_prob[lv])
                    target = (
                        dot_student_prob[lv].new(dot_student_prob[lv].size(0)).fill_(1)
                    )
                    loss += cos_loss_func(
                        dot_student_prob[lv], dot_teacher_prob[lv], target
                    )
                    loss = 0.5 * loss / (lv + 1)
                    # Perform a backward pass to calculate the gradients.
                    loss.backward(retain_graph=True)
                    total_train_loss += loss.item()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                for lv in distillers:
                    for d in lv:
                        clip_grad_norm_(d.student.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                for lv in optimizers:
                    for optimizer in lv:
                        optimizer.step()

            # Calculate the average loss over all of the batches.
            avg_distill_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            print("  Average training loss: {0:.2f}".format(avg_distill_loss))
            print(f"Distillation time: {time() - t0:.2f}s")

            def flatten(l):
                return [item for sublist in l for item in sublist]

        flt_nodes = flatten(nodes)
        flt_distillers = flatten(distillers)
        # save the model
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for node, distiller in zip(flt_nodes, flt_distillers):
            torch.save(
                distiller.student.state_dict(),
                f"{save_dir}/self-distill_bert_{node.name}.pt",
            )
            node.model = distiller.student

        print(f"Self-training time: {time() - t0:.2f}s")

        print(f"Making result ...")
        outputs = []
        for step, batch in enumerate(train_dataloader):
            # Progress update.
            if step % batch_size == 0 and not step == 0:
                # Report progress.
                print("  Batch {:>5,}  of  {:>5,}.".format(step, len(train_dataloader)))

            batch_input_ids = batch[0].to(self.device)
            batch_input_mask = batch[1].to(self.device)

            all_logits = [
                [get_teacher_logits(node.model) for node in level] for level in nodes
            ]
            dot_prob = probabilities_dot_product(all_logits, batch_size)
            outputs.extend(dot_prob[-1].argmax(1).tolist())
        # return global_classifier
        return outputs  # y_pred
