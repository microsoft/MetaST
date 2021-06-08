"""BERT NER Inference.""" 

from __future__ import absolute_import, division, print_function

import json
import os

import torch
import torch.nn.functional as F
from nltk import word_tokenize
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
#from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy

from transformers import BertTokenizer


class Ner:
    def __init__(self,model_dir: str):
        self.model , self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["id2label"]
        self.max_seq_length = 64
        self.label_map = {int(k):v for k,v in self.label_map.items()}
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "config.json"):
        model_config = os.path.join(model_dir,model_config)
        model_config = json.load(open(model_config))
        output_config_file = os.path.join(model_dir, CONFIG_NAME)
        output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = BertForTokenClassification.from_pretrained(model_dir)
        model.load_state_dict(torch.load(output_model_file,map_location='cpu'))#To map from CPU?
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """ tokenize input"""
        #words = word_tokenize(text) # it will split message? into message ?
        words = text.split()
        #print (words)
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            print (word, token)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        print (tokens)
        ## insert "[CLS]"
        tokens.insert(0,"[CLS]")
        ## insert "[SEP]"
        tokens.append("[SEP]")
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        '''
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        '''
        
        
        return input_ids,input_mask,segment_ids,valid_positions

    def predict(self, text: str):
        input_ids,input_mask,segment_ids,valid_positions = self.preprocess(text)
        input_ids = torch.tensor([input_ids],dtype=torch.long)
        input_mask = torch.tensor([input_mask],dtype=torch.long)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long)

        #print (input_ids)
        #print (input_mask)
        #print (segment_ids)
        #print (valid_positions)

    
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask)
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        #shape [batch_size,sequence length]
        logits_label = logits_label.detach().cpu().numpy()
        # import ipdb; ipdb.set_trace()
        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label[0])]

        #extract valid inputs
        logits_label = [logits_label[0][index] for index,i in enumerate(input_mask[0]) if i.item()==1]
        # Remove [CLS] and [SEP]
        # Ref: https://github.com/kamalkraj/BERT-NER/issues/9
        logits_label.pop(0) 
        logits_label.pop()
        

        

        assert len(logits_label) == len(valid_positions)
        labels = []
        for valid,label in zip(valid_positions,logits_label):
            if valid:
                labels.append(self.label_map[label])
        #words = word_tokenize(text)
        words = text.split()
        assert len(labels) == len(words)
        #output = {word:{"tag":label,"confidence":confidence} for word,label,confidence in zip(words,labels,logits_confidence)}

        # use tuple instead of dictionary
        output = [[word,label,confidence] for word,label,confidence in zip(words,labels,logits_confidence)]
        return output



class compact_model(Ner):
    
    def __init__(self, model_dir: str, args):
        Ner.__init__(self, model_dir)
        #load student model 
        from Modelings import SimpleRNN

        self.model = SimpleRNN(model_dir,args)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'student_model_no_crf.bin'),map_location='cpu'))
        

    def predict(self, text: str):
        #self.tokenizer = BertTokenizer.from_pretrained(model_config["bert_model"],do_lower_case=True)
        input_ids,input_mask,segment_ids,valid_positions = self.preprocess(text)
        input_ids = torch.tensor([input_ids],dtype=torch.long)
        input_mask = torch.tensor([input_mask],dtype=torch.long)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long)

        print (input_ids)
        #print (input_mask)
        #print (segment_ids)
        print (valid_positions)

    
        with torch.no_grad():
            #model_prediction
            logits = self.model(input_ids)
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        #shape [batch_size,sequence length]
        logits_label = logits_label.detach().cpu().numpy()
        # import ipdb; ipdb.set_trace()
        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label[0])]

        #extract valid inputs
        logits_label = [logits_label[0][index] for index,i in enumerate(input_mask[0]) if i.item()==1]
        # Remove [CLS] and [SEP]
        # Ref: https://github.com/kamalkraj/BERT-NER/issues/9
        logits_label.pop(0) 
        logits_label.pop()
        

        

        assert len(logits_label) == len(valid_positions)
        labels = []
        for valid,label in zip(valid_positions,logits_label):
            if valid:
                labels.append(self.label_map[label])
            print (self.label_map[label])
        #words = word_tokenize(text)
        #print (l)
        words = text.split()
        
        assert len(labels) == len(words)
        output = {word:{"tag":label,"confidence":confidence} for word,label,confidence in zip(words,labels,logits_confidence)}


        #allow duplicate
        '''
        from collections import defaultdict
        output = defaultdict(list)
        for word,label,confidence in zip(words,labels,logits_confidence): 
            output[word].append({"tag":label,"confidence":confidence})
            #output[word].append({"tag":label,"confidence":confidence})

        print (words)
        print (output)
        '''
        
        return output


class compact_model_crf(Ner):
    
    def __init__(self, model_dir: str, args):
        Ner.__init__(self, model_dir)
        #load student model 
        from Modelings import BiLSTM_CRF

        self.model = BiLSTM_CRF(model_dir,args)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'student_model.bin'),map_location='cpu'))
        

    def predict(self, text: str):
        #self.tokenizer = BertTokenizer.from_pretrained(model_config["bert_model"],do_lower_case=True)
        input_ids,input_mask,segment_ids,valid_positions = self.preprocess(text)
        input_ids = torch.tensor([input_ids],dtype=torch.long)
        input_mask = torch.tensor([input_mask],dtype=torch.long)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long)

        print (input_ids)
        #print (input_mask)
        #print (segment_ids)
        print (valid_positions)

    
        with torch.no_grad():
            #model_prediction
            #update for crf
            _,logits_label,logits = self.model(input_ids,input_mask)
            logits_label = numpy.array(logits_label)
            #print (logits_label)
        logits = F.softmax(logits,dim=2)
        #logits_label = torch.argmax(logits,dim=2)
        #shape [batch_size,sequence length]
        #logits_label = logits.detach().cpu().numpy()
        # import ipdb; ipdb.set_trace()
        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label[0])]

        #extract valid inputs
        logits_label = [logits_label[0][index] for index,i in enumerate(input_mask[0]) if i.item()==1]
        # Remove [CLS] and [SEP]
        # Ref: https://github.com/kamalkraj/BERT-NER/issues/9
        logits_label.pop(0) 
        logits_label.pop()
        

        

        assert len(logits_label) == len(valid_positions)
        labels = []
        for valid,label in zip(valid_positions,logits_label):
            if valid:
                labels.append(self.label_map[label])
            #print (self.label_map[label])
        #words = word_tokenize(text)
        #print (l)
        words = text.split()
        
        assert len(labels) == len(words)
        output = {word:{"tag":label,"confidence":confidence} for word,label,confidence in zip(words,labels,logits_confidence)}


        #allow duplicate
        '''
        from collections import defaultdict
        output = defaultdict(list)
        for word,label,confidence in zip(words,labels,logits_confidence): 
            output[word].append({"tag":label,"confidence":confidence})
            #output[word].append({"tag":label,"confidence":confidence})

        print (words)
        print (output)
        '''
        
        return output


def parser_config(parser):

    parser.add_argument("--embedding_size",
                        default=300,
                        type=int,
                        required=False,
                        help="The word embedding size of student network")
    parser.add_argument("--hidden_units",
                        default=300,
                        type=int,
                        required=False,
                        help="The hidden units for student network")
    parser.add_argument("--num_labels",
                        default=59,
                        type=int,
                        required=False,
                        help="The size of label set. The default one for communication ")
    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        required=False,
                        help="batch_size for inference ")
    parser.add_argument("--seq_length",
                        default=32,
                        type=int,
                        required=False,
                        help="MaxSeq len for inference")
    parser.add_argument("--teacher_model_path",
                        default='.',
                        type=str,
                        required=False,
                        help="Path to bert config")
    parser.add_argument("--encoder_type",
                        default='LSTM',
                        type=str,
                        required=False,
                        help="Path to bert config")



    return parser.parse_args()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    args = parser_config(parser)


    model = Ner("comm_out/")
    print (model.load_model("comm_out/"))
    print (dir(compact_model))

    student_model = compact_model("comm_out/",args)
    print (dir(compact_model))
    #print (student_model.load_model("out/"))
    input_text = "hey cortana please send this message to haoda that I don't know"
    
    print (student_model.model)
    output = student_model.predict(input_text)
    print (output)