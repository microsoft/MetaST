from typing import NamedTuple
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_inference import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import namedtuple
import argparse
from  crf_new import CRF
import os
import json


class SimpleRNN(torch.nn.Module):


    def __init__(self,teacher_model_path,args):
        super(SimpleRNN, self).__init__()
        self.model_config = self.load_model_config("config.json",teacher_model_path)
        self.input_size = args.embedding_size
        self.lstm_hidden = args.hidden_units 
        self.lstm_layer = 1
        self.dropout = nn.Dropout(0.2)
        self.bilstm_flag = True
        #TODO:batch_first true is not supported by onnx
        if args.encoder_type == 'LSTM':
            self.lstm = nn.LSTM(self.input_size,  self.lstm_hidden, num_layers=self.lstm_layer, batch_first=False, bidirectional=self.bilstm_flag)
        elif args.encoder_type == 'GRU':
            self.lstm = nn.GRU(self.input_size,  self.lstm_hidden, num_layers=self.lstm_layer, batch_first=False, bidirectional=self.bilstm_flag)
        else:
            self.lstm = nn.RNN(self.input_size,  self.lstm_hidden, num_layers=self.lstm_layer, batch_first=False, bidirectional=self.bilstm_flag)
        #self.lstm = nn.LSTM(self.input_size,  self.lstm_hidden, num_layers=self.lstm_layer, batch_first=False, bidirectional=self.bilstm_flag)
        self.model_object = Ner(teacher_model_path)
        self.hidden2tag = nn.Linear(2*self.lstm_hidden, len(self.load_model_config("config.json",teacher_model_path).id2label))


        
        import copy
        '''     
        self.embed = copy.deepcopy(self.model_object.model.bert.embeddings)
        self.embed.word_embeddings.weight.requires_grad = True
        self.embed.position_embeddings.weight.requires_grad = True
        #self.embed.token_type_embeddings = None
        print (self.embed.word_embeddings.weight)
        '''

        #TODO remove token embedding through copy

        self.embed = Embeddings(self.model_config)
        self.embed.word_embeddings.weight = copy.deepcopy(self.model_object.model.bert.embeddings.word_embeddings.weight)
        self.embed.position_embeddings.weight = copy.deepcopy(self.model_object.model.bert.embeddings.position_embeddings.weight)
        self.embed.LayerNorm = copy.deepcopy(self.model_object.model.bert.embeddings.LayerNorm)


    def load_model_config(self,model_conifg,teacher_model_path):
        '''
        load teacher model hyperparameters
        '''

        model_config = os.path.join(teacher_model_path, 'config.json')
        model_config = json.load(open(model_config))
        #convert dictionary to namedtuple
        def convert(dictionary):
            return namedtuple('GenericDict', dictionary.keys())(**dictionary)
        model_config = convert(model_config)
        print (model_config)

        return model_config


    def forward(self, input_ids, mode='training'):
        #Todo: add mask inside module
        #replace the padding one with bert input
        #with input_id, input_mask, segment_ids as inputs
        #TODO: reverse input dimension as onnx does not support batch_first

        #[batch_size,seq_len,word_dim]
        word_representation = self.embed(input_ids)
        word_representation = word_representation[:,:,0:self.input_size]
        

        #reverse
        word_representation = word_representation.transpose(0, 1)  # Swaps 2nd and 1st dimension
        #word_representation = self.dropout(word_representation)
        

        hidden = None
        lstm_out, hidden = self.lstm(word_representation, hidden)

        #reverse again
        lstm_out = lstm_out.transpose(0, 1)  # Swaps 2nd and 1st dimension
       
        outputs = self.hidden2tag(lstm_out)

        
        ##for onnx export
        if mode == 'onnx-export':
            outputs = torch.argmax(F.log_softmax(outputs, dim=2), dim=2)
            print (mode)

        return outputs







#Reuse the embeddings module in BERT
# ref: https://github.com/huggingface/pytorch-transformers/blob/0a74c88ac609c03293c69b61cfa7c9b084e38cdb/pytorch_transformers/modeling_distilbert.py#L531
class Embeddings(nn.Module):
    def __init__(self,
                 config):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        #self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        '''
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.max_position_embeddings,
                                         dim=config.dim,
                                         out=self.position_embeddings.weight)
        '''

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.
        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) # (max_seq_length)
        #position_ids = torch.arange(seq_length, dtype=torch.int32, device=input_ids.device) # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)                      # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)                   # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)        # (bs, max_seq_length, dim)
        #token_type_embeddings = self.token_type_embeddings(token_type_ids)

        #TODO: we can remove postion embeddings in CNN 
        #embeddings = word_embeddings + position_embeddings + token_type_embeddings  # (bs, max_seq_length, dim)
        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)             # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)               # (bs, max_seq_length, dim)
        return embeddings




class BiLSTM_CRF(nn.Module):
    def __init__(self, teacher_model_path, args):
        super().__init__()
        self.lstm = SimpleRNN(teacher_model_path,args)
        
        self.crf = CRF(55,False)

    def forward(self, x, mask=None):
        emissions = self.lstm(x)
        #print (emissions.shape)
        scores, tag_seq = self.crf._viterbi_decode(emissions, mask)
        #print (scores.shape,tag_seq.shape)
        return scores, tag_seq ,emissions

    def loss(self, x, y, mask=None):
        emissions = self.lstm(x)
        nll = self.crf.neg_log_likelihood_loss(emissions, mask, y)
        return nll







def config(parser):

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



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = config(parser)
    
    teacher_model_path = "comm_out/"

    #model = Simple1DCNN(cfg).double()
    #model = Simple1DCNN().double()
    model = SimpleRNN(teacher_model_path,args)

    teacher_model = Ner("comm_out/")
    #the embedding size
    #print (teacher_model.model.bert.embeddings.word_embeddings)
    text = "Share my presentation with Cortana"
    input_ids,input_mask,segment_ids,valid_positions = teacher_model.preprocess(text)
    input_ids = torch.tensor([input_ids],dtype=torch.long)
    input_mask = torch.tensor([input_mask],dtype=torch.long)
    segment_ids = torch.tensor([segment_ids],dtype=torch.long)

    
    #print (model(input_ids).shape)
    
    

    model = SimpleRNN(teacher_model_path,args)
    print (model)
    print (model(input_ids).shape)


    '''
    model = BiLSTM_CRF(teacher_model_path,args)
    

    # Check predictions before training
    print('Predictions before training:')
    with torch.no_grad():
        scores, seqs, logits = model(input_ids)
        print (numpy.array(seqs).shape)
        for score, seq in zip(scores, seqs):
            #str_seq = " ".join(ids_to_tags(seq, ix_to_tag))
            print('%.2f: %s' % (score.item(), seq))
    '''