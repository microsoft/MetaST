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
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig,
                                              BertForTokenClassification)
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

from data_utils import data_processors
from data_utils.data_processors import *
from data_utils.torch2onnx import torch2onnx

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
    parser.add_argument("--student_model_dir",
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
                        default=128,
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
    parser.add_argument("--do_continual_training",
                        action='store_true',
                        help="Whether to continual training on Mustpass set")
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
                        type=int,
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

def train_epoch(model, train_examples, label_list,tokenizer,num_train_optimization_steps,device,n_gpu,optimizer,args,processor):
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

    model.train()

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    eval_results = [] 
    loss_list = []

    early_stopping = EarlyStopping(patience=5, verbose=True)

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            s_logits = model(input_ids)

            #Define loss function
            mlm_loss_fct = nn.CrossEntropyLoss(reduction='none')
            ce_loss_fct = nn.KLDivLoss(reduction='batchmean')        
            
            active_loss = input_mask.view(-1) == 1
            active_logits = s_logits.view(-1, s_logits.size(-1))[active_loss]  #[batch_size*seq_len, label_size] good for select,token
            active_labels = label_ids.view(-1)[active_loss]

            #print(active_logits.shape,active_labels.shape)
            #print (active_logits)
            #print(active_labels,label_ids)
            
            if args.crf:
                loss = model.loss(input_ids, label_ids, mask=input_mask)
            else:
                #loss = mlm_loss_fct(active_logits, active_labels)
                input_mask_double = input_mask.double()
                loss = (mlm_loss_fct(s_logits.view(-1, 57), label_ids.view(-1))*input_mask_double.view(-1)).sum()

                print(active_logits.shape,active_labels.shape,active_loss.shape)
                logger.info("  Current Training loss = %f", loss.mean())
                logger.info("  Current Training loss = %f", loss.sum())
                logger.info("  Current Training loss = %f", loss.item())

                print(active_labels,label_ids)

            


            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()


            tr_loss += loss.item()
            logger.info("  Current Training loss = %d", loss.item())
            loss_list.append(tr_loss)
            

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
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
        #track eval for each epoch
        eval_result =  eval_epoch(model, model, label_list, tokenizer, device, args,processor,'validation')
        eval_results.append(eval_result)
        model.train() # reenable model train



        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(-1*eval_result, model) #update for F1 score
        if early_stopping.early_stop:
            print("Early stopping")
            break
        logger.info("  Current Evaluation results = %d", eval_result)

        
    print (eval_results)
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    #with open(output_config_file, 'w') as f:
    #    f.write(model_to_save.config.to_json_string())
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    model_config = {"bert_model": "bert-base-uncased", "do_lower": args.do_lower_case
        , "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1
        , "label_map": label_map}
    json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))

    output_student_model_file = os.path.join(args.output_dir, "student_model_new.bin")
    torch.save(model.state_dict(), output_student_model_file)
    

    
    #track azureml log
    best_val_f1 = eval_epoch(model, model, label_list, tokenizer, device, args,processor,'validation')
    run.log('best_val_f1', np.float(best_val_f1))
    #run.log_list("training_loss", tr_loss, description='loss of mustpass')


def eval_epoch(student_model,model,label_list,tokenizer,device,args,processor,mode):

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
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    print (label_map)


    #Add student model evaluation
    student_model.eval()
    

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)


        ##resolve dynamic padding issue
        ##for batch size 1
        #input_ids = input_ids[input_mask!=0]
        #input_ids = input_ids.view(1,input_ids.size(-1))
        print (input_ids)
        print (input_mask)

        with torch.no_grad():
            #plus argmax to decode
            if args.crf:
                _, paths,_ = student_model(input_ids,input_mask)
                #student_logits = np.array(paths)
                student_logits = paths.detach().cpu().numpy()
            else:
                student_logits = student_model(input_ids)
            logits = student_logits
        if not args.crf:
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2) ##cahnge to softmax
            logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        
        
        for i, mask in enumerate(input_mask):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(mask):
                if j == 0:
                    continue
                if m:
                    #Ref: https://github.com/kamalkraj/BERT-NER/issues/13 to remove 0 (which is the pad label, not in the list though)
                    # candidate solutions like add it to label list or resolve it in runtime
                    #if label_map[label_ids[i][j]] != "X" and logits[i][j]!=0:   #TODO: hacky solution for now align the training part
                        #print (label_ids[i][j],logits[i][j])                                                      #the bert label starts from 1,hacky solution for now
                    if label_map[label_ids[i][j]] != "X" and logits[i][j]!=0:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])
                        print (label_map[label_ids[i][j]],logits[i][j])                                                               
                else:
                    print (temp_1)
                    print (temp_2)
                    temp_1.pop()
                    temp_2.pop()
                    break
            y_true.append(temp_1)
            y_pred.append(temp_2)
            assert len(temp_1)==len(temp_2)
    report = classification_report(y_true, y_pred, digits=4)


    eval_file_name = mode + "_eval_results_students.txt"
    output_eval_file = os.path.join(args.output_dir, eval_file_name)
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results students*****")
        logger.info("\n%s", report)
        writer.write(report)

    f1 = f1_score(y_true, y_pred)
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
    num_labels = len(label_list) + 1

    tokenizer = transformers.tokenization_bert.BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case,do_basic_tokenize=args.do_basic_tokenize)
    


    train_examples = None
    num_train_optimization_steps = None
    if args.do_continual_training:
        train_examples = processor.get_eval_examples(args.train_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
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
    output_config_file = os.path.join(args.teacher_model_path, CONFIG_NAME)
    output_model_file = os.path.join(args.teacher_model_path, WEIGHTS_NAME)
    config = BertConfig(output_config_file)
    model = BertForTokenClassification(config, num_labels=num_labels)
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
    print(student_model)
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
    #param_optimizer = list(student_model.named_parameters())
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
        #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_proportion*num_train_optimization_steps, t_total=num_train_optimization_steps)  # PyTorch scheduler
        #scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_proportion*num_train_optimization_steps, t_total=num_train_optimization_steps)  # PyTorch scheduler
        #scheduler = get_cosine_schedule_with_warmup(optimizer,args.warmup_proportion*num_train_optimization_steps,num_train_optimization_steps)

        optimizer = optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate, momentum=0.9)
        #optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    #load student model 
    output_student_model_file = os.path.join(args.student_model_dir, 'student_model.bin')
        
    if not torch.cuda.is_available():
        student_model.load_state_dict(torch.load(output_student_model_file,map_location='cpu'))
    else:
        student_model.load_state_dict(torch.load(output_student_model_file))

    model.to(device)
    student_model.to(device) #AttributeError: 'IncompatibleKeys' object has no attribute 'to'

    if args.do_continual_training:

        #train_epoch(student_model,model, train_examples,unsupervised_train_dataset, label_list, tokenizer, num_train_optimization_steps,device,n_gpu,optimizer,args,processor,scheduler)
        train_epoch(student_model, train_examples, label_list,tokenizer,num_train_optimization_steps,device,n_gpu,optimizer,args,processor)
    

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        #eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,'validation')
        eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,'test')
        eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,'test_generated')
        eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,'test_generated_no_contact')
        eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,'target_set')

        ##convert pytorch model to onnx format
        onnx_output_path = os.path.join(args.output_dir, "cortana_communication_enus_MV4.slots.rnn.onnx.bin")
        student_model.eval()
        torch2onnx(student_model,onnx_output_path)
        eval_epoch(student_model,model, label_list, tokenizer, device, args,processor,'validation')

        


if __name__ == "__main__":
    main()