from torch.nn.functional import gelu, elu
import torch.nn.functional as F
import torch.nn as nn
import math
import torch
from collections import OrderedDict
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import pdb
from torch.nn import CrossEntropyLoss, MSELoss

def functional_bert(fast_weights, config, input_ids=None, attention_mask=None, token_type_ids=None,
                    position_ids=None, head_mask=None, inputs_embeds=None, labels=None, is_train=True, args=None, reduce=True, use_focal_loss=None,
                    sentence_loss=None):
    encoder_hidden_states = None
    encoder_extended_attention_mask = None


    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if config.is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            causal_mask = causal_mask.to(torch.long)
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape,
                                                                                                    attention_mask.shape))
    extended_attention_mask = extended_attention_mask.to(
        dtype=next((p for p in fast_weights.values())).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=next((p for p in fast_weights.values())).dtype)
    else:
        head_mask = [None] * config.num_hidden_layers

    embedding_output = functional_embeeding(fast_weights, config, input_ids, position_ids,
                                            token_type_ids, inputs_embeds, is_train=is_train)

    encoder_outputs = functional_encoder(fast_weights, config, embedding_output,
                                         attention_mask=extended_attention_mask,
                                         head_mask=head_mask, encoder_hidden_states=encoder_hidden_states,
                                         encoder_attention_mask=encoder_extended_attention_mask, is_train=is_train)

    sequence_output = encoder_outputs

    sequence_output = F.dropout(sequence_output, p=config.hidden_dropout_prob)
    logits = functional_classifier(fast_weights, sequence_output)

    outputs = (F.softmax(logits), embedding_output)

    num_labels = logits.size(-1)

    if labels is not None:


        if not use_focal_loss:
            loss_fct = CrossEntropyLoss(reduce=reduce)
        else:
            loss_fct = FocalLoss(gamma=args.gamma, reduce=reduce)
        # Only keep active parts of the loss

        if sentence_loss is not None:

            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            loss = loss.view(labels.size(0), labels.size(1))
            if sentence_loss == 'mean':
                num = (loss > 0).sum(-1)
                loss = loss.sum(-1)
                loss = loss / num
            elif sentence_loss == 'max':
                loss, _ = loss.max(-1)
        elif attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_loss = active_loss & (labels.view(-1) > 0)
            active_logits = logits.view(-1, config.num_labels)[active_loss]
            # active_labels = torch.where(
            #     active_loss, labels.view(-1), torch.tensor(CrossEntropyLoss().ignore_index).type_as(labels)
            # )
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        outputs = (loss,) + outputs


    #pdb.set_trace()
    return outputs


def functional_embeeding(fast_weights, config, input_ids, position_ids,
                         token_type_ids, inputs_embeds=None, is_train=True):
    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]

    seq_length = input_shape[1]
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    if inputs_embeds is None:
        inputs_embeds = F.embedding(input_ids, fast_weights['bert.embeddings.word_embeddings.weight'], padding_idx=0)

    position_embeddings = F.embedding(position_ids, fast_weights['bert.embeddings.position_embeddings.weight'])
    token_type_embeddings = F.embedding(token_type_ids, fast_weights['bert.embeddings.token_type_embeddings.weight'])

    embeddings = inputs_embeds + position_embeddings + token_type_embeddings

    embeddings = F.layer_norm(embeddings, [config.hidden_size],
                              weight=fast_weights['bert.embeddings.LayerNorm.weight'],
                              bias=fast_weights['bert.embeddings.LayerNorm.bias'],
                              eps=config.layer_norm_eps)

    embeddings = F.dropout(embeddings, p=config.hidden_dropout_prob, training=is_train)

    return embeddings


def transpose_for_scores(config, x):
    new_x_shape = x.size()[:-1] + (config.num_attention_heads, int(config.hidden_size / config.num_attention_heads))
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

def functional_classifier(fast_weights, sequence_output):

    logits = F.linear(sequence_output,
                          fast_weights['classifier.weight'],
                          fast_weights['classifier.bias'])
    return logits

def functional_self_attention(fast_weights, config, layer_idx,
                              hidden_states, attention_mask, head_mask,
                              encoder_hidden_states, encoder_attention_mask,
                              is_train=True):
    attention_head_size = int(config.hidden_size / config.num_attention_heads)
    all_head_size = config.num_attention_heads * attention_head_size

    mixed_query_layer = F.linear(hidden_states,
                                 fast_weights['bert.encoder.layer.' + layer_idx + '.attention.self.query.weight'],
                                 fast_weights['bert.encoder.layer.' + layer_idx + '.attention.self.query.bias'])

    if encoder_hidden_states is not None:
        mixed_key_layer = F.linear(encoder_hidden_states,
                                   fast_weights['bert.encoder.layer.' + layer_idx + '.attention.self.key.weight'],
                                   fast_weights['bert.encoder.layer.' + layer_idx + '.attention.self.key.bias'])
        mixed_value_layer = F.linear(encoder_hidden_states,
                                     fast_weights['bert.encoder.layer.' + layer_idx + '.attention.self.value.weight'],
                                     fast_weights['bert.encoder.layer.' + layer_idx + '.attention.self.value.bias'])
        attention_mask = encoder_attention_mask
    else:
        mixed_key_layer = F.linear(hidden_states,
                                   fast_weights['bert.encoder.layer.' + layer_idx + '.attention.self.key.weight'],
                                   fast_weights['bert.encoder.layer.' + layer_idx + '.attention.self.key.bias'])
        mixed_value_layer = F.linear(hidden_states,
                                     fast_weights['bert.encoder.layer.' + layer_idx + '.attention.self.value.weight'],
                                     fast_weights['bert.encoder.layer.' + layer_idx + '.attention.self.value.bias'])

    query_layer = transpose_for_scores(config, mixed_query_layer)
    key_layer = transpose_for_scores(config, mixed_key_layer)
    value_layer = transpose_for_scores(config, mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    if is_train:
        attention_probs = F.dropout(attention_probs, p=config.attention_probs_dropout_prob)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = context_layer
    return outputs


def functional_out_attention(fast_weights, config, layer_idx,
                             hidden_states, input_tensor,
                             is_train=True):
    hidden_states = F.linear(hidden_states,
                             fast_weights['bert.encoder.layer.' + layer_idx + '.attention.output.dense.weight'],
                             fast_weights['bert.encoder.layer.' + layer_idx + '.attention.output.dense.bias'])

    hidden_states = F.dropout(hidden_states, p=config.hidden_dropout_prob, training=is_train)
    hidden_states = F.layer_norm(hidden_states + input_tensor, [config.hidden_size],
                                 weight=fast_weights[
                                     'bert.encoder.layer.' + layer_idx + '.attention.output.LayerNorm.weight'],
                                 bias=fast_weights[
                                     'bert.encoder.layer.' + layer_idx + '.attention.output.LayerNorm.bias'],
                                 eps=config.layer_norm_eps)

    return hidden_states


def functional_attention(fast_weights, config, layer_idx,
                         hidden_states, attention_mask=None, head_mask=None,
                         encoder_hidden_states=None, encoder_attention_mask=None,
                         is_train=True):
    self_outputs = functional_self_attention(fast_weights, config, layer_idx,
                                             hidden_states, attention_mask, head_mask,
                                             encoder_hidden_states, encoder_attention_mask, is_train)

    attention_output = functional_out_attention(fast_weights, config, layer_idx,
                                                self_outputs, hidden_states, is_train)
    return attention_output


def functional_intermediate(fast_weights, config, layer_idx, hidden_states, is_train=True):
    weight_name = 'bert.encoder.layer.' + layer_idx + '.intermediate.dense.weight'
    bias_name = 'bert.encoder.layer.' + layer_idx + '.intermediate.dense.bias'
    hidden_states = F.linear(hidden_states, fast_weights[weight_name], fast_weights[bias_name])
    hidden_states = gelu(hidden_states)

    return hidden_states


def functional_output(fast_weights, config, layer_idx, hidden_states, input_tensor, is_train=True):
    hidden_states = F.linear(hidden_states,
                             fast_weights['bert.encoder.layer.' + layer_idx + '.output.dense.weight'],
                             fast_weights['bert.encoder.layer.' + layer_idx + '.output.dense.bias'])

    hidden_states = F.dropout(hidden_states, p=config.hidden_dropout_prob, training=is_train)
    hidden_states = F.layer_norm(hidden_states + input_tensor, [config.hidden_size],
                                 weight=fast_weights['bert.encoder.layer.' + layer_idx + '.output.LayerNorm.weight'],
                                 bias=fast_weights['bert.encoder.layer.' + layer_idx + '.output.LayerNorm.bias'],
                                 eps=config.layer_norm_eps)
    return hidden_states


def functional_layer(fast_weights, config, layer_idx, hidden_states, attention_mask,
                     head_mask, encoder_hidden_states, encoder_attention_mask, is_train=True):
    self_attention_outputs = functional_attention(fast_weights, config, layer_idx,
                                                  hidden_states, attention_mask, head_mask,
                                                  encoder_hidden_states, encoder_attention_mask, is_train)

    attention_output = self_attention_outputs
    intermediate_output = functional_intermediate(fast_weights, config, layer_idx, attention_output, is_train)
    layer_output = functional_output(fast_weights, config, layer_idx,
                                     intermediate_output, attention_output, is_train)

    return layer_output


def functional_encoder(fast_weights, config, hidden_states, attention_mask,
                       head_mask, encoder_hidden_states, encoder_attention_mask, is_train=True):
    for i in range(0, config.num_hidden_layers):
        layer_outputs = functional_layer(fast_weights, config, str(i),
                                         hidden_states, attention_mask, head_mask[i],
                                         encoder_hidden_states, encoder_attention_mask, is_train)
        hidden_states = layer_outputs

    outputs = hidden_states
    return outputs

def update_parameters(model, loss, step_size=0.5, first_order=False):
    """Update the parameters of the model, with one step of gradient descent.
    Parameters
    ----------
    model : `MetaModule` instance
        Model.
    loss : `torch.FloatTensor` instance
        Loss function on which the gradient are computed for the descent step.
    step_size : float (default: `0.5`)
        Step-size of the gradient descent step.
    first_order : bool (default: `False`)
        If `True`, use the first-order approximation of MAML.
    Returns
    -------
    params : OrderedDict
        Dictionary containing the parameters after one step of adaptation.
    """
    #pdb.set_trace()

    grads = torch.autograd.grad(loss, model.parameters(),
        create_graph=not first_order, allow_unused=True)
    #pdb.set_trace()

    params = OrderedDict()
    i = 0
    import pdb

    for name, param in model.named_parameters():
        if 'pooler' in name:
            i += 1
            continue

        name = name.replace("module.", "")
        try:
            if param.requires_grad:
                params[name] = param - step_size * grads[i]
                i += 1
            else:
                params[name] = param
        except:
            pdb.set_trace()

    return params