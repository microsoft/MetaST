# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
from sampler import UncertaintyDatasetSampler
import torch.nn.functional as F

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from model import RobertaForTokenClassification_v2, BertForTokenClassification
from data_utils import load_and_cache_examples, get_labels
from model_utils import multi_source_label_refine, soft_frequency,  opt_grad
from eval import evaluate
from modeling_meta_bert import update_parameters, functional_bert
from data_utils import split_data

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter



logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification_v2, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def teacher_initialize(args, model_class, config, model_folder, learning_rate, t_total):
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, \
                      eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))

    if args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    model.zero_grad()
    return model, optimizer, scheduler


def initialize(args, model_class, config, t_total):

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, \
                      eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    if args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    # Check if saved optimizer or scheduler states exist

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    model.zero_grad()
    return model, optimizer, scheduler


def update_sampling_weight(args,
                           model,
                           eval_dataset,
                           meta_sampler):

    eval_sampler = RandomSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)
    model.eval()
    for step, eval_batch in enumerate(eval_dataloader):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)
        eval_inputs = {"input_ids": eval_batch[0], "attention_mask": eval_batch[1],
                       "labels": eval_batch[3]}

        outputs = model(**eval_inputs)
        loss, logits, final_embeds = outputs[0], outputs[1], outputs[2]
        meta_sampler.async_update_matrix(eval_batch[-1], eval_batch[3], logits)
    model.train()


def meta_selection(args,
                   config,
                   model,
                   optimizer,
                   input_list,
                   meta_dataloader,
                   sampling_step=1):

    if sampling_step < 0:
        sampling_step = len(meta_dataloader)
    la_inputs = input_list[0]
    un_inputs = input_list[1]

    for i in range(sampling_step):

        eval_batch_nocuda = next(meta_dataloader.__iter__())
        eval_batch = tuple(t.to(args.device) for t in eval_batch_nocuda)
        eval_inputs = {"input_ids": eval_batch[0], "attention_mask": eval_batch[1],
                       "labels": eval_batch[3]}
        self_training_fast_model = model
        self_training_fast_model.eval()

        outputs = self_training_fast_model(**un_inputs, reduce=False)
        loss, logits, final_embeds = outputs[0], outputs[1], outputs[2]
        active_loss = outputs[-1]

        if args.use_labeled_data == 1:
            la_outputs = self_training_fast_model(**la_inputs)
            la_loss = la_outputs[0]
            if args.n_gpu > 1:
                la_loss = la_loss.mean()

        elif args.use_labeled_data == 2:
            la_outputs = self_training_fast_model(**la_inputs, reduce=False)
            la_loss = la_outputs[0]

        else:
            la_loss = 0

        if args.self_training_label_mode == "hard":

            if args.use_token_weight == 1:
                if args.use_labeled_data == 2:
                    combined_loss = torch.cat([loss, la_loss], dim=0)
                    weight = torch.zeros(combined_loss.size(), requires_grad=True).to(args.device)
                    new_loss = (combined_loss * weight).sum()
                else:
                    weight = torch.zeros(loss.size(), requires_grad=True).to(args.device)
                    new_loss = (loss * weight).sum() + args.labeled_beta * la_loss
            else:
                weight = torch.zeros(logits.size(0), requires_grad=True).to(args.device)
                sen_weight = weight.unsqueeze(-1).repeat(1, logits.size(1))
                sen_weight = sen_weight.view(-1)[active_loss]
                new_loss = (loss * sen_weight).sum() + args.labeled_beta * la_loss

        else:
            weight = torch.zeros(loss.sum(-1).size(), requires_grad=True).to(args.device)
            new_loss = (loss.sum(-1) * weight).sum() + args.labeled_beta * la_loss

        try:
            fast_weights = update_parameters(self_training_fast_model, new_loss, step_size=args.controler_step_size)
        except:
            pdb.set_trace()
        outputs = functional_bert(fast_weights, config, **eval_inputs, args=args)
        loss, logits, final_embeds = outputs[0], outputs[1], outputs[2]


        if args.n_gpu > 1:
            loss = loss.mean()

        grad_eps = torch.autograd.grad(loss, weight, only_inputs=True)[0]

        # Line 11 computing and normalizing the weights
        if i == 0:
            loss_weight = -grad_eps
        else:
            loss_weight -= grad_eps
    self_training_fast_model.zero_grad()

    loss_weight = torch.clamp(loss_weight, min=0)

    if args.use_token_weight != 1:
        loss_weight = loss_weight.unsqueeze(-1).repeat(1, logits.size(1))
        loss_weight = loss_weight.view(-1)[active_loss]

    norm_c = torch.sum(loss_weight) + 0.0000001

    if norm_c != 0:
        loss_weight = loss_weight / norm_c
    else:
        loss_weight = loss_weight

    model.train()

    return loss.data, loss_weight.detach()


def finetune_model(args, config, model_folder, model, optimizer, scheduler, train_dataloader, model_class, tokenizer,
                   labels, pad_token_label_id, learning_rate, best_test, best_dev, global_step,
                   t_total, tb_writer, logger, steps, eval_dataloader=None,  test_dataloader=None):
    logger.info("Start fituning.... ")
    if model_folder is not None:
        model, optimizer, scheduler = teacher_initialize(args, model_class, config, model_folder, learning_rate,
                                                         t_total=t_total)

    for i in tqdm(range(steps)):

        batch = next(train_dataloader.__iter__())
        batch = tuple(t.to(args.device) for t in batch)

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        outputs = model(**inputs, args=args)
        loss = outputs[0]
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        if (i + 1) % args.logging_steps == 0:
            results, _, best_dev, dev_updated, c_result = evaluate(args, model, tokenizer,
                                                                   labels,
                                                                   pad_token_label_id, best_dev,
                                                                   mode="dev", data_loader=eval_dataloader,
                                                                   prefix='dev [Step {}/{} ]'.format(
                                                                       global_step, t_total),
                                                                   verbose=False, final=False)
            results, _, best_test, test_updated, c_result = evaluate(args, model, tokenizer,
                                                                     labels,
                                                                     pad_token_label_id, best_test,
                                                                     mode="test", data_loader=test_dataloader,
                                                                     prefix='test [Step {}/{} ]'.format(
                                                                         global_step, t_total),
                                                                     verbose=False, final=False)

            logger.info(
                "***** Global_step: %d, finetune_step: %d *****", \
                global_step, i)

            for key, value in results.items():
                tb_writer.add_scalar("test_{}".format(key), value, global_step)
                logger.info("test %s, %.4f", key, value)



            output_dirs = []
            if args.local_rank in [-1, 0] and dev_updated:
                output_dirs.append(os.path.join(args.output_dir, "dev-best"))

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dirs.append(os.path.join(args.output_dir, "checkpoint-{}".format(global_step)))

            if len(output_dirs) > 0:
                for output_dir in output_dirs:
                    logger.info("Saving model checkpoint to %s", args.output_dir)
                    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                    # They can then be reloaded using `from_pretrained()`
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
                    with open(output_test_results_file, "w", encoding="utf-8") as writer:
                        for key in sorted(results.keys()):
                            writer.write("{} = {}\n".format(key, str(results[key])))
                        writer.write(c_result)
    logger.info("Finished fituning.... ")
    return best_dev, best_test


def train(args, model_class, config, tokenizer, labels,
          pad_token_label_id):
    """ Train the model """

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'tfboard'))

    train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
    unlabeled_train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id,
                                                      mode="unlabeled_train")
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="dev", final=False)

    test_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="test", final=False)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.unlabeled_train_batch_size = args.per_gpu_unlabeled_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size  = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    unlabeled_train_sampler = RandomSampler(unlabeled_train_dataset) if args.local_rank == -1 else DistributedSampler(
        unlabeled_train_dataset)
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, sampler=unlabeled_train_sampler,
                                            batch_size=args.unlabeled_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(
        eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                            batch_size=args.eval_batch_size)

    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(
        test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=args.eval_batch_size)


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    elif args.self_training_begin_epoch > 0:

        self_training_start_epoch = args.self_training_begin_epoch

        args.self_training_begin_step = args.self_training_begin_epoch * len(
            train_dataloader) * args.gradient_accumulation_steps
        t_total = args.self_training_begin_step
        t_total += (len(unlabeled_train_dataloader) // args.gradient_accumulation_steps * (
                args.num_train_epochs - self_training_start_epoch))

    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        self_training_start_epoch = args.self_training_begin_step // (
                len(train_dataloader) // args.gradient_accumulation_steps)
        t_total += (len(unlabeled_train_dataloader) // args.gradient_accumulation_steps * (
                args.num_train_epochs - self_training_start_epoch))

    model, optimizer, scheduler = initialize(args, model_class, config, args.self_training_begin_step)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    softmax = torch.nn.Softmax(dim=1)

    tr_loss, logging_loss = 0.0, 0.0
    best_dev, best_test, tmp_dev = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    loss = 0


    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    set_seed(args)  # Added here for reproductibility

    for epoch in train_iterator:

        if global_step >= args.self_training_begin_step:
            break

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # Update labels periodically after certain begin step

            if global_step >= args.self_training_begin_step:
                break

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )

            outputs = model(**inputs, args=args)

            loss, logits, final_embeds = outputs[0], outputs[1], outputs[2]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:

                        logger.info("***** Global_step: %d, Entropy loss: %.4f*****", \
                                    global_step, loss )

                        results, _, best_dev, dev_updated, _ = evaluate(args, model, tokenizer, labels,
                                                                        pad_token_label_id,
                                                                        best_dev, mode="dev", data_loader=eval_dataloader,
                                                                        prefix='dev [Step {}/{} | Epoch {}/{}]'.format(
                                                                            global_step, t_total, epoch,
                                                                            args.num_train_epochs),
                                                                        verbose=False, final=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        results, _, best_test, test_updated, c_result = evaluate(args, model, tokenizer, labels,
                                                                                 pad_token_label_id, best_test,
                                                                                 mode="test", data_loader=test_dataloader,
                                                                                 prefix='test [Step {}/{} | Epoch {}/{}]'.format(
                                                                                     global_step, t_total, epoch,
                                                                                     args.num_train_epochs),
                                                                                 verbose=False, final=False)

                        for key, value in results.items():
                            tb_writer.add_scalar("test_{}".format(key), value, global_step)

                        output_dirs = []
                        if args.local_rank in [-1, 0] and dev_updated:
                            output_dirs.append(os.path.join(args.output_dir, "dev-best"))

                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                            output_dirs.append(os.path.join(args.output_dir, "checkpoint-{}".format(global_step)))

                        if len(output_dirs) > 0:
                            for output_dir in output_dirs:
                                logger.info("Saving model checkpoint to %s", args.output_dir)
                                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                                # They can then be reloaded using `from_pretrained()`
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)
                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", output_dir)
                            output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
                            with open(output_test_results_file, "w", encoding="utf-8") as writer:
                                for key in sorted(results.keys()):
                                    writer.write("{} = {}\n".format(key, str(results[key])))
                                writer.write(c_result)

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    ################Self-training###############

    self_training_hp_label = args.self_training_hp_label

    if args.per_gpu_meta_batch_size != -1:
        meta_batch_size = args.per_gpu_meta_batch_size
    else:
        meta_batch_size = len(meta_dataset)

    if args.per_pgu_labeled_batch_size != -1:
        labeled_batch_size = args.per_pgu_labeled_batch_size
    else:
        labeled_batch_size = len(labeled_dataset)

    train_features = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode='train',
                                             raw_feature=True)
    args.use_labeled_data = 0

    labeled_dataset, meta_dataset = split_data(train_features, args.split_ratio, args.split_mode)

    if len(labeled_dataset) != 0:
        labeled_sampler = RandomSampler(labeled_dataset) if args.local_rank == -1 else DistributedSampler(
            labeled_dataset)
        labeled_dataloader = DataLoader(labeled_dataset, sampler=labeled_sampler, batch_size=labeled_batch_size)

    if len(meta_dataset) != 0:
        meta_sampler = UncertaintyDatasetSampler(meta_dataset, smoothness_type=args.smoothness_type,
                                                 mode=args.sampling_strategy)
        meta_dataloader = DataLoader(meta_dataset, sampler=meta_sampler, batch_size=meta_batch_size)

    labeled_beta = args.labeled_beta

    steps_trained_in_current_epoch = global_step - args.self_training_begin_step

    if global_step >= args.self_training_begin_step:

        self_training_teacher_model = copy.deepcopy(model)
        model_folder = os.path.join(args.output_dir, "dev-best")
        self_training_teacher_model.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))
        self_training_teacher_model.eval()

        for step in tqdm(range(args.self_training_begin_step, args.max_steps)):
            labeled_beta = labeled_beta * args.labeled_beta_decay

            # # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            if args.use_labeled_data != 0:

                try:
                    labeled_batch = next(labeled_dataloader)
                except:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    labeled_batch = next(labeled_dataloader.__iter__())
                # labeled_batch = next(labeled_dataloader.__iter__())
                labeled_batch = tuple(t.to(args.device) for t in labeled_batch)

            try:
                unlabeled_batch = next(unlabeled_train_dataloader)
            except:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                unlabeled_batch = next(unlabeled_train_dataloader.__iter__())

            # unlabeled_batch = next(unlabeled_train_dataloader.__iter__())
            unlabeled_batch = tuple(t.to(args.device) for t in unlabeled_batch)

            model.train()

            # Update a new teacher periodically
            delta = global_step - args.self_training_begin_step

            if delta % args.self_training_period == 0:
                best_folder = os.path.join(args.output_dir, "dev-best")
                if global_step != args.self_training_begin_step:
                    teacher_leraning_rate = args.learning_rate * 0.1
                    model_folder = os.path.join(args.output_dir, "tmp-best")
                    if args.finetune == 1:
                        best_dev, best_test = finetune_model(args, config, model_folder, model, optimizer, scheduler,
                                                             train_dataloader, model_class, tokenizer, labels,
                                                             pad_token_label_id, teacher_leraning_rate, best_test, best_dev,
                                                             global_step, t_total, tb_writer, logger,
                                                             steps=args.self_training_begin_step,
                                                             eval_dataloader=eval_dataloader,
                                                             test_dataloader=test_dataloader)

                labeled_beta = args.labeled_beta
                self_training_teacher_model, teacher_optimizer, teacher_scheduler = teacher_initialize(args,
                                                                                                       model_class,
                                                                                                       config,
                                                                                                       best_folder,
                                                                                                       args.learning_rate * 0.1,
                                                                                                       t_total)

                self_training_teacher_model.eval()

                results, _, best_test, is_updated, c_result = evaluate(args, self_training_teacher_model, tokenizer,
                                                                       labels,
                                                                       pad_token_label_id, best_test,
                                                                       mode="test", data_loader=test_dataloader,
                                                                       prefix='test [Step {}/{} ]'.format(
                                                                           global_step, t_total),
                                                                       verbose=False, final=False)
                train_features = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode='train',
                                                         raw_feature=True)

                labeled_dataset, meta_dataset = split_data(train_features, args.split_ratio, args.split_mode)

                if len(labeled_dataset) != 0:
                    labeled_sampler = RandomSampler(labeled_dataset) if args.local_rank == -1 else DistributedSampler(
                        labeled_dataset)
                    labeled_dataloader = DataLoader(labeled_dataset, sampler=labeled_sampler,
                                                    batch_size=labeled_batch_size)
                if len(meta_dataset) != 0:
                    meta_sampler = UncertaintyDatasetSampler(meta_dataset, smoothness_type=args.smoothness_type, mode=args.sampling_strategy)
                    meta_dataloader = DataLoader(meta_dataset, sampler=meta_sampler, batch_size=meta_batch_size)

                if args.self_training_reinit:
                    tmp_dev = [0, 0, 0]
                    model, optimizer, scheduler = initialize(args, model_class, config, t_total)


            if delta > args.sampling_begin_step:
                if delta % args.sampling_update_freq == 0 and len(meta_dataset) != 0:
                    if args.sampling_strategy != 'uniform':
                        update_sampling_weight(args, model, meta_dataset, meta_sampler)


            # Using current teacher to update the label
            unlabeled_inputs = {"input_ids": unlabeled_batch[0], "attention_mask": unlabeled_batch[1]}
            if args.use_labeled_data != 0:
                labeled_inputs = {"input_ids": labeled_batch[0], "attention_mask": labeled_batch[1],
                                  "labels": labeled_batch[3]}
            else:
                labeled_inputs = []


            with torch.no_grad():
                outputs = self_training_teacher_model(**unlabeled_inputs, args=args)


            if args.self_training_label_mode == "hard":
                pred_labels = torch.argmax(outputs[2], axis=2)
                pred_labels, label_mask = multi_source_label_refine(args, unlabeled_batch[5], unlabeled_batch[3],
                                                                    pred_labels,
                                                                    pad_token_label_id, self_training_hp_label,
                                                                    pred_logits=outputs[0])
            elif args.self_training_label_mode == "soft":
                pred_labels = soft_frequency(logits=outputs[0], power=2)
                pred_labels, label_mask = multi_source_label_refine(args, unlabeled_batch[5], unlabeled_batch[3],
                                                                    pred_labels,
                                                                    pad_token_label_id, self_training_hp_label)
            elif args.self_training_label_mode == "logits":
                logits = outputs[0]
                y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
                pred_labels, label_mask = multi_source_label_refine(args, unlabeled_batch[5], unlabeled_batch[3], y,
                                                                    pad_token_label_id, self_training_hp_label)

            unlabeled_inputs = {"input_ids": unlabeled_batch[0], "attention_mask": unlabeled_batch[1],
                                "labels": pred_labels}

            if args.use_psuedo_data_selection:
                unlabeled_outputs = model(**unlabeled_inputs, reduce=False, args=args)
            else:
                unlabeled_outputs = model(**unlabeled_inputs, args=args)

            un_loss, logits, final_embeds = unlabeled_outputs[0], unlabeled_outputs[1], unlabeled_outputs[
                2]  # model outputs are always tuple in pytorch-transformers (see doc)

            mt_loss, vat_loss, meta_loss, labeled_loss = 0, 0, 0, 0

            if args.use_labeled_data == 1:
                labeled_outputs = model(**labeled_inputs, args=args)
                labeled_loss, logits, final_embeds = labeled_outputs[0], labeled_outputs[1], labeled_outputs[2]
            elif args.use_labeled_data == 2:
                labeled_outputs = model(**labeled_inputs, reduce=False, args=args)
                labeled_loss, logits, final_embeds = labeled_outputs[0], labeled_outputs[1], labeled_outputs[2]

            if args.use_psuedo_data_selection:

                meta_loss, loss_weight = meta_selection(args, config, model, optimizer,
                                                              [labeled_inputs, unlabeled_inputs],
                                                              meta_dataloader,
                                                              sampling_step=args.controler_sampling_steps)

                if args.self_training_label_mode != "hard":
                    un_loss = un_loss.sum(-1)

                if args.use_labeled_data == 2:
                    combined_loss = torch.cat([un_loss, labeled_loss], dim=0)
                    loss = (combined_loss * loss_weight).sum()
                else:
                    loss = (un_loss * loss_weight).sum()

            if args.use_labeled_data != 2:
                loss = loss + args.labeled_beta * labeled_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        logger.info(
                            "***** Global_step: %d, Entropy loss: %.4f Meta Loss is %.4f *****", \
                            global_step, loss, meta_loss)

                        results, _, tmp_dev, tmp_updated, _ = evaluate(args, model, tokenizer, labels,
                                                                       pad_token_label_id,
                                                                       tmp_dev, mode="dev", data_loader=eval_dataloader,
                                                                       prefix='dev [Step {}/{} ]'.format(
                                                                           global_step, t_total),
                                                                       verbose=False, final=False)
                        if results['f1'] > best_dev[-1]:
                            dev_updated = True
                            best_dev = copy.deepcopy(tmp_dev)
                        else:
                            dev_updated = False

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        results, _, best_test, test_updated, c_result = evaluate(args, model, tokenizer, labels,
                                                                                 pad_token_label_id, best_test,
                                                                                 mode="test", data_loader=test_dataloader,
                                                                                 prefix='test [Step {}/{}]'.format(
                                                                                     global_step, t_total),
                                                                                 verbose=False, final=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("test_{}".format(key), value, global_step)
                            logger.info("test %s, %.4f", key, value)

                        output_dirs = []
                        if args.local_rank in [-1, 0] and dev_updated:
                            output_dirs.append(os.path.join(args.output_dir, "dev-best"))

                        if args.local_rank in [-1, 0] and tmp_updated:
                            output_dirs.append(os.path.join(args.output_dir, "tmp-best"))

                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                            output_dirs.append(os.path.join(args.output_dir, "checkpoint-{}".format(global_step)))

                        if len(output_dirs) > 0:
                            for output_dir in output_dirs:
                                logger.info("Saving model checkpoint to %s", args.output_dir)
                                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                                # They can then be reloaded using `from_pretrained()`
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)
                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", output_dir)
                                output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
                                with open(output_test_results_file, "w", encoding="utf-8") as writer:
                                    for key in sorted(results.keys()):
                                        writer.write("{} = {}\n".format(key, str(results[key])))
                                    writer.write(c_result)

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model, global_step, tr_loss / global_step, best_dev, best_test


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--is_json", action="store_true", help="Using json or txt.")
    parser.add_argument("--n_shot", default=5, type=int,
                        help="number of shot.")
    parser.add_argument("--dataset", default='', type=str,
                        help="name of dataset.")

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_unlabeled_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--controler_step_size", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=5e-6, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="BETA1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float, help="BETA2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=200,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=20, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--scheduler", type=str, default="consine", help="scheduler: linear or consine.")

    parser.add_argument('--labeled_beta', default=0, type=float, help="coefficient of labeled_loss term.")
    parser.add_argument('--per_pgu_labeled_batch_size', default=1, type=int, help="labeled data batch size.")
    parser.add_argument('--labeled_beta_decay', default=0.999, type=float, help="decay loss term.")

    parser.add_argument('--split_ratio', default=0, type=float, help="labeled data : meta data.")
    parser.add_argument('--split_mode', default='random', type=str, help="coefficient of labeled_loss term.")

    # Meta-selector
    parser.add_argument('--use_psuedo_data_selection', action="store_true", help='use selection or not.')
    parser.add_argument('--controler_sampling_steps', type=int, default=1, help='the controler sampling steps.')
    parser.add_argument('--per_gpu_meta_batch_size', type=int, default=8, help='use product_metrics or not.')
    parser.add_argument('--use_labeled_data', type=int, default=0,
                        help='0: dont use labeled data, 1: use labeled data as loss, 2: need to select labeled data.')
    parser.add_argument('--use_token_weight', type=int, default=0,
                        help='0: use sentence weight, 1: use token weight.')
    parser.add_argument('--use_labeled_loss', action="store_true", help='Use labeledl loss.')
    parser.add_argument('--sampling_strategy', type=str, default='uniform',
                        help='loss_var, loss_decay, logits, uniform.')
    parser.add_argument('--smoothness_type', type=str, default='mean',
                        help='mean, max.')
    parser.add_argument('--finetune', type=int, default=1,
                        help='0: no finetune, 1 finetune.')

    # self-training
    parser.add_argument('--self_training_reinit', type=int, default=0,
                        help='re-initialize the student model if the teacher model is updated.')
    parser.add_argument('--self_training_begin_step', type=int, default=4000,
                        help='the begin step (usually after the first epoch) to start self-training.')
    parser.add_argument('--self_training_begin_epoch', type=int, default=-1,
                        help='the begin epoch (usually after the first epoch) to start self-training.')
    parser.add_argument('--self_training_label_mode', type=str, default="hard",
                        help='pseudo label type. choices:[hard(default), soft].')
    parser.add_argument('--self_training_period', type=int, default=3000, help='the self-training period.')
    parser.add_argument('--self_training_hp_label', type=float, default=6.0, help='use high precision label.')
    parser.add_argument('--self_training_ensemble_label', type=int, default=0, help='use ensemble label.')
    parser.add_argument('--self_student_period', type=int, default=3000, help='the self-training period.')
    parser.add_argument('--sampling_begin_step', type=int, default=1000, help='the self-training period.')
    parser.add_argument('--sampling_update_freq', type=int, default=10, help='the self-training period.')

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    labels = get_labels(args.data_dir)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    best_dev = None
    if args.do_train:
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        model, global_step, tr_loss, best_dev, best_test = train(args,  model_class, config, tokenizer, labels,
                                                                 pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving last-practice: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        if not best_dev:
            best_dev = [0, 0, 0]
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _, best_dev, _, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_dev,
                                                 mode="dev",  prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w", encoding="utf-8") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model_path = os.path.join(args.output_dir, "dev-best")

        model = model_class.from_pretrained(model_path)
        model.to(args.device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        #if not best_test:
        best_test = [0, 0, 0]
        result, predictions, _, _, c_result = evaluate(args, model, tokenizer, labels, pad_token_label_id,
                                                       best=best_test, mode="test")
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")

        with open(output_test_results_file, "w", encoding="utf-8") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

            writer.write(c_result)
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w", encoding="utf-8") as writer:
            if args.is_json:
                with open(os.path.join(args.data_dir, "test.json"), "r") as f:
                    example_id = 0
                    data = json.load(f)
                    for item in data:
                        output_line = str(item["str_words"]) + " " + predictions[example_id].pop(0) + "\n"
                        writer.write(output_line)
                        example_id += 1
            else:
                with open(os.path.join(args.data_dir, "test.txt"), "r", encoding="utf-8") as f:
                    example_id = 0
                    for line in f:
                        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                            writer.write(line)
                            if not predictions[example_id]:
                                example_id += 1
                        elif predictions[example_id]:
                            output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                            writer.write(output_line)
                        else:
                            logger.warning(
                                "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0]
                            )
    return results


if __name__ == "__main__":
    main()
