from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys
import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
# from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
#                                               BertConfig,
#                                               BertForTokenClassification)

from transformers import (
    WEIGHTS_NAME,
    CONFIG_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
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
import pdb
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from transformers.optimization import *
#from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
#from pytorch_pretrained_bert.tokenization import BertTokenizer
import transformers
from transformers import BertTokenizer
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import data_processors
from data_processors import *
from torch2onnx import torch2onnx

from Modelings import *
from itertools import cycle
from EarlyStopping import *
from azureml.core.run import Run
run = Run.get_context()



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)




def train_config(parser):
    ## Required parameters
    parser.add_argument("--unsupervised_train_corpus",
                        default="lm_data/train.txt",
                        type=str,
                        required=True,
                        help="The input of unsupervised train corpus.")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--teacher_model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The teacher model dir. ")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--fine_train_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--valid_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input valid dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--test_generated_dir",
                        default='valid.txt',
                        type=str,
                        required=True,
                        help="The input synthetic set. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--test_generated_no_contact_dir",
                        default='test.txt',
                        type=str,
                        required=True,
                        help="The input synthetic set dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--target_set_dir",
                        default='target_set.txt',
                        type=str,
                        required=True,
                        help="The input target_set dir. Should contain the .tsv files (or other data files) for the task.")          

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--alpha",
                        default=1,
                        type=int,
                        help="Task weight for student network")
    parser.add_argument("--beta",
                        default=1,
                        type=float,
                        help="Task weight for unsupervised objective ")
    parser.add_argument("--temperature",
                        default=1,
                        type=int,
                        help="Task weight for student network")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--do_basic_tokenize",
                        action='store_true',
                        help="Set this flag if you are using basic tokenzier.")

    ##for student model
    parser.add_argument("--embedding_size",
                        default=768,
                        type=int,
                        required=False,
                        help="The word embedding size of student network")
    parser.add_argument("--hidden_units",
                        default=600,
                        type=int,
                        required=False,
                        help="The hidden units for student network")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        required=False,
                        help="The weight_decay for student network")
    parser.add_argument("--encoder_type",
                        default="RNN",
                        type=str,
                        help="The encoder type")
    parser.add_argument("--crf",
                        action='store_true',
                        help="Whether to add crf")
    

    return parser.parse_args()

def train_epoch(student_model,model, train_examples, fine_examples, unsupervised_train_dataset, label_list,tokenizer,num_train_optimization_steps,device,n_gpu,optimizer,args,processor,scheduler):
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data) #shuffle the data
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    #Add unlabeled data here
    unsupervised_train_sampler = RandomSampler(unsupervised_train_dataset)
    unsupervised_train_dataloader = DataLoader(unsupervised_train_dataset, sampler=unsupervised_train_sampler, batch_size=args.train_batch_size*10)
    logger.info("  Num examples of unlabeled data = %d", len(unsupervised_train_dataset))

    fine_features = convert_examples_to_features(
        fine_examples, label_list, args.max_seq_length, tokenizer)
    fine_all_input_ids = torch.tensor([f.input_ids for f in fine_features], dtype=torch.long)
    fine_all_input_mask = torch.tensor([f.input_mask for f in fine_features], dtype=torch.long)
    fine_all_segment_ids = torch.tensor([f.segment_ids for f in fine_features], dtype=torch.long)
    fine_all_label_ids = torch.tensor([f.label_id for f in fine_features], dtype=torch.long)
    fine_train_data = TensorDataset(fine_all_input_ids, fine_all_input_mask, fine_all_segment_ids, fine_all_label_ids)
    fine_train_sampler = RandomSampler(fine_train_data)
    fine_train_dataloader = DataLoader(fine_train_data, sampler=fine_train_sampler, batch_size=args.train_batch_size)



    #add student model here
    student_model.train()
    model.eval()

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    eval_results = []

    early_stopping = EarlyStopping(patience=5, verbose=True)
    early_stopping_condition = False

    temperature = args.temperature
    alpha = args.alpha
    beta = args.beta

    end = -1


    for _ in trange(int(args.num_train_epochs), desc="Epoch"):


        if early_stopping.early_stop:
            print("Early stopping")
            end += 1
            if end >= 1:
                break
            train_dataloader = fine_train_dataloader
            early_stopping.early_stop = False
            alpha = 0

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        #modify for smaller size list
        #for step, (batch,unsupervised_batch) in enumerate(tqdm(zip(train_dataloader,cycle(unsupervised_train_dataloader)), desc="Iteration")):
        #update for larger unlabeled data
        for step, (batch,unsupervised_batch) in enumerate(tqdm(zip(cycle(train_dataloader),unsupervised_train_dataloader), desc="Iteration")):
        #for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)

            #handle unsupervised data and generate unsupervised loss
            unsupervised_input_ids, unsupervised_input_mask, unsupervised_segment_ids= unsupervised_batch
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                unsupervised_input_ids = unsupervised_input_ids.cuda()
                unsupervised_input_mask = unsupervised_input_mask.cuda()
                unsupervised_segment_ids = unsupervised_segment_ids.cuda()
                student_model.cuda()


            #add student model here
            #Get logits and use kd loss
            with torch.no_grad():               
                t_logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
                t_logits_unsupervised = model(input_ids=unsupervised_input_ids,
                                              token_type_ids=unsupervised_segment_ids,
                                              attention_mask=unsupervised_input_mask)[0]

            if args.crf:
                _, _, s_logits = student_model(input_ids=input_ids,attention_mask=input_mask) #TODO:
                _, _, s_logits_unsupervised = student_model(input_ids=unsupervised_input_ids,attention_mask=unsupervised_input_mask)
            else:         
                s_logits = student_model(input_ids)
                s_logits_unsupervised = student_model(unsupervised_input_ids)



            assert t_logits.shape == s_logits.shape
            assert t_logits_unsupervised.shape == s_logits_unsupervised.shape

            #Define loss function
            mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss_fct = nn.KLDivLoss(reduction='batchmean')        
            
            #student_loss = mlm_loss_fct(s_logits.view(-1, s_logits.size(-1)), label_ids.view(-1))
            active_loss = label_ids.view(-1) > 0
            active_logits = s_logits.view(-1, s_logits.size(-1))[active_loss]
            active_labels = label_ids.view(-1)[active_loss]

            #print(active_logits.shape,active_labels.shape)
            #print (active_logits)
            #print(active_labels,label_ids)
            
            if args.crf:
                student_loss = student_model.loss(input_ids, label_ids, mask=input_mask)
            else:
                student_loss = mlm_loss_fct(active_logits, active_labels)

            

            
            #TODO: better module

            #convert to booltensor/bytetensor, to adjust for masked select
            mask = input_mask.unsqueeze(-1).expand_as(s_logits)   # (bs, seq_lenth, voc_size)
            mask =  torch.tensor(mask,dtype=torch.uint8,device=input_ids.device)


            #print (t_logits.shape,s_logits.shape,mask.shape)
            #print (mask)

            s_logits_slct = torch.masked_select(s_logits, mask==1)            # (bs * seq_length * voc_size) modulo the 1s in mask
            s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))      # (bs * seq_length, voc_size) modulo the 1s in mask
            t_logits_slct = torch.masked_select(t_logits, mask==1)            # (bs * seq_length * voc_size) modulo the 1s in mask
            t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))      # (bs * seq_length, voc_size) modulo the 1s in mask
            assert t_logits_slct.size() == s_logits_slct.size()

            #Supervised distillation loss
            loss_ce = ce_loss_fct(F.log_softmax(s_logits_slct/temperature, dim=-1),
                                   F.softmax(t_logits_slct/temperature, dim=-1)) * (temperature)**2



            mask = unsupervised_input_mask.unsqueeze(-1).expand_as(s_logits_unsupervised)   # (bs, seq_lenth, voc_size)
            mask =  torch.tensor(mask,dtype=torch.uint8,device=input_ids.device)

            s_logits_slct_unsupervised = torch.masked_select(s_logits_unsupervised,  mask==1)            # (bs * seq_length * voc_size) modulo the 1s in mask
            s_logits_slct_unsupervised = s_logits_slct_unsupervised.view(-1, s_logits_unsupervised.size(-1))      # (bs * seq_length, voc_size) modulo the 1s in mask
            t_logits_slct_unsupervised = torch.masked_select(t_logits_unsupervised,  mask==1)            # (bs * seq_length * voc_size) modulo the 1s in mask
            t_logits_slct_unsupervised = t_logits_slct_unsupervised.view(-1, s_logits_unsupervised.size(-1))      # (bs * seq_length, voc_size) modulo the 1s in mask
            assert t_logits_slct_unsupervised.size() == s_logits_slct_unsupervised.size()

            #unsupervised distillation loss
            loss_ce_unsupervised = ce_loss_fct(F.log_softmax(s_logits_slct_unsupervised/temperature, dim=-1),
                                   F.softmax(t_logits_slct_unsupervised/temperature, dim=-1)) * (temperature)**2

            #combine the loss function

            #loss = alpha * loss.mean() + (1-alpha)*loss_ce + alpha * student_loss.mean()
            loss =  loss_ce + beta * loss_ce_unsupervised + alpha * student_loss.mean()
            #loss =   beta * loss_ce_unsupervised + alpha * student_loss.mean()
            #loss =  loss_ce_unsupervised
            #loss =  loss_ce + alpha * student_loss
            #loss =   alpha * student_loss #Trained from scratch


            if step % 100 ==0:


                # if step == 0:
                    # eval_result = eval_epoch(model, model, label_list, tokenizer, device, args, processor, early_stopping,
                    #                          'validation', role='teacher')
                    # print(eval_result)




                eval_result =  eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,early_stopping, 'validation')
                student_model.train()
                model.eval()
                eval_results.append(eval_result)
                print (eval_results)

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                early_stopping(-1*eval_result, student_model) #update for F1 score

            

            print (loss.item(),loss_ce.item(),loss_ce_unsupervised.item(),student_loss.item())
            

            #for rnn and multi task we should do it before loss backward?
            #optimizer.zero_grad()

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps


            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            
            tr_loss += loss.item()
            #print (input_ids.grad,s_logits.grad)

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step

                #Todo: add clip_grad_norm
                max_grad_norm = 1
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()
                global_step += 1

        '''
        #track eval for each epoch
        eval_result =  eval_epoch(student_model,model, label_list, tokenizer, device, args,processor)
        eval_results.append(eval_result)
        model.train() # reenable model train
        student_model.train()
        logger.info("  Current Evaluation results = %d", eval_result)
        '''

        

        
    print (eval_results)

    # Save a trained model and the associated configuration

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())
    label_map = {i: label for i, label in enumerate(label_list, 1)}

    model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case
        , "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1
        , "label_map": label_map}
    json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))


    #save student model

    # load the last checkpoint with the best model
    student_model.load_state_dict(torch.load('checkpoint.pt'))

    output_student_model_file = os.path.join(args.output_dir, "student_model.bin")
    torch.save(student_model.state_dict(), output_student_model_file)


    #track azureml log
    # best_val_f1 = eval_epoch(student_model,model, label_list, tokenizer, device, args,processor, e'validation')
    # run.log('best_val_f1', np.float(best_val_f1))




def eval_epoch(student_model,model,label_list,tokenizer,device,args,processor,early_stop, mode, role='student'):

    if mode=='test':
        eval_examples = processor.get_test_examples(args.data_dir)
    elif mode=='test_generated':
        eval_examples = processor.get_eval_examples(args.test_generated_dir)
    elif mode=='test_generated_no_contact':
        eval_examples =processor.get_eval_examples(args.test_generated_no_contact_dir)
    elif mode=='target_set':
        eval_examples = processor.get_eval_examples(args.target_set_dir)
    elif mode == 'validation':
        eval_examples = processor.get_eval_examples(args.valid_dir)

    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list)}
    #print (label_map)


    #Add student model evaluation
    student_model.eval()

    

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            #add student model
            #plus argmax to decode
            if args.crf:
                _, paths,_ = student_model(input_ids,input_mask)
                #student_logits = np.array(paths)
                student_logits = paths.detach().cpu().numpy()
            else:
                if role == 'teacher':
                    student_logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
                else:
                    student_logits = student_model(input_ids)
            logits = student_logits
        if not args.crf:
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        
        for i, mask in enumerate(input_mask):
            temp_1 = []
            temp_2 = []
            for j,_ in enumerate(mask):

                if label_ids[i][j]>=0 and label_map[label_ids[i][j]] != "X":   #TODO: hacky solution for now align the training part
                                                       #the bert label starts from 1,hacky solution for now
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])

            y_true.append(temp_1)
            y_pred.append(temp_2)
            assert len(temp_1)==len(temp_2)
    report = classification_report(y_true, y_pred, digits=4)


    eval_file_name = mode + "_eval_results_students.txt"
    f1 = f1_score(y_true, y_pred)
    if early_stop.best_score is not None and f1 >= early_stop.best_score:
        output_eval_file = os.path.join(args.output_dir, eval_file_name)
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results students*****")
            logger.info("\n%s", report)
            writer.write(report)


    return f1




def main():
    parser = argparse.ArgumentParser()
    args =train_config(parser)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {"ner" :data_processors.NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels(args.data_dir)

    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_path, do_lower_case=args.do_lower_case)


    #num_train_optimization_steps in optimizer

    #"Loading Unsupervised Train Dataset", args.unsupervised_train_corpus)
    unsupervised_train_dataset = BERTDataset(args.unsupervised_train_corpus, tokenizer, seq_len=args.max_seq_length,
                corpus_lines=None, on_memory=args.on_memory)


    train_examples = None
    fine_train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        #train_examples = processor.get_train_examples(args.data_dir)
        train_examples = processor.get_eval_examples(args.train_dir)
        fine_train_examples = processor.get_eval_examples(args.fine_train_dir)

        num_train_optimization_steps = int(
            (len(train_examples)+len(unsupervised_train_dataset) + len(fine_train_examples)) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()


    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    '''
    model = BertForTokenClassification.from_pretrained(args.bert_model,
                                                       cache_dir=cache_dir,
                                                       num_labels = num_labels)
    '''
    #Load finetuned model from  dir
    output_config_file = os.path.join(args.teacher_model_path, 'config.json')
    output_model_file = os.path.join(args.teacher_model_path, WEIGHTS_NAME)
    config = BertConfig.from_pretrained(args.teacher_model_path)
    model = BertForTokenClassification.from_pretrained(args.teacher_model_path)
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(output_model_file,map_location='cpu'))
    else:
        model.load_state_dict(torch.load(output_model_file))

    
    if args.fp16:
        model.half()
    model.to(device)

    #add student model here
    if args.crf:
        student_model = BiLSTM_CRF(args.teacher_model_path,args)
    else:
        student_model = SimpleRNN(args.teacher_model_path,args)
    #print(student_model)
    student_model.to(device)


    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError \
                ("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    #add student model parameters
    ##TODO: named_parameters and parameter
    #param_optimizer = list(model.named_parameters())+list(student_model.named_parameters())
    param_optimizer = list(model.named_parameters())+list(student_model.named_parameters())

    for ele in param_optimizer:
        n,p = ele
        print (n, p.shape)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        #{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError \
                ("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        #TODO: Ablation study for optimizer
        #change to transformer implementation
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_proportion*num_train_optimization_steps, t_total=num_train_optimization_steps)  # PyTorch scheduler
        #scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_proportion*num_train_optimization_steps, t_total=num_train_optimization_steps)  # PyTorch scheduler
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        # )
        scheduler = get_cosine_schedule_with_warmup(optimizer,args.warmup_proportion*num_train_optimization_steps,num_train_optimization_steps)


    if args.do_train:
        '''
        freeze the bert layer 
        '''

        '''
        for param in model.bert.parameters():
            param.requires_grad = False


        for name,param in model.bert.named_parameters():
            if 'embeddings' not in name:
                param.requires_grad = False
        '''

        #To do: better module
        train_epoch(student_model,model, train_examples, fine_train_examples, unsupervised_train_dataset, label_list, tokenizer, num_train_optimization_steps,device,n_gpu,optimizer,args,processor,scheduler)

        # Load a trained model and config that you have fine-tuned
    else:
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = BertForTokenClassification(config, num_labels=num_labels)
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(output_model_file,map_location='cpu'))
        else:
            model.load_state_dict(torch.load(output_model_file))


    #load student model from previous output
    output_student_model_file = os.path.join(args.output_dir, 'student_model.bin')
        
    #print(torch.load(output_student_model_file,map_location='cpu'))
    #student_model = student_model.load_state_dict(torch.load(output_student_model_file))
    if not torch.cuda.is_available():
        student_model.load_state_dict(torch.load(output_student_model_file,map_location='cpu'))
    else:
        student_model.load_state_dict(torch.load(output_student_model_file))

    model.to(device)
    student_model.to(device) #AttributeError: 'IncompatibleKeys' object has no attribute 'to'

    

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,'test')
        eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,'test_generated')
        eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,'test_generated_no_contact')
        eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,'target_set')

        ##convert pytorch model to onnx format
        onnx_output_path = os.path.join(args.output_dir, "cortana_communication_enus_MV4.slots.rnn.onnx.bin")
        torch2onnx(student_model,onnx_output_path)

        


if __name__ == "__main__":
    main()