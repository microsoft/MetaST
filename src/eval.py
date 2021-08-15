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
import pdb
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from data_utils import load_and_cache_examples, tag_to_id, get_chunks
from flashtool import Logger
import os
import pickle

logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer, labels, pad_token_label_id, best, mode, data_loader=None, prefix="", verbose=True, final=True):

    if data_loader is not None:
        eval_dataloader = data_loader
    else:
        eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, final=final)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)

        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation %s *****", prefix)
    if verbose:
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()

    i = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        i += 1
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    out_id_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_id_list = [[] for _ in range(out_label_ids.shape[0])]
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                preds_list[i].append(label_map[preds[i][j]])
                out_id_list[i].append(out_label_ids[i][j])
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_id_list[i].append(preds[i][j])


    p = precision_score(out_label_list, preds_list)
    r = recall_score(out_label_list, preds_list)
    new_F = f1_score(out_label_list, preds_list)
    c_result = classification_report(out_label_list, preds_list)

    results = {}

    is_updated = False
    if new_F > best[-1]:
        best = [p, r, new_F]
        is_updated = True

    results.update({
        "loss": eval_loss,
        "precision": p,
        "recall": r,
        "f1": new_F,
        "best_precision": best[0],
        "best_recall": best[1],
        "best_f1": best[-1]
    })

    logger.info("***** Eval results %s *****",  prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    model.train()

    return results, preds_list, best, is_updated, c_result


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index and label_ids[i, j] not in start_end_label_id:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list

