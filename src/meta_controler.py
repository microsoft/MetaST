import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.nn.functional import gelu, elu

class VNet(nn.Module):
    def __init__(self, config, args):
        super(VNet, self).__init__()
        hidden = 16
        self.num_labels = config.num_labels
        self.linear1 = nn.Linear(self.num_labels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(768, hidden)

        self.args = args
        if args.use_gumbel:
            self.linear3 = nn.Linear(hidden, 2)
        else:
            self.linear3 = nn.Linear(hidden, 1)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                label_mask=None,
                logits = None,
                final_emb=None,
                tau=1,
                is_loss=False):

        #pdb.set_trace()


        x1 = self.linear1(logits)
        x1 = gelu(x1)
        if final_emb is not None:
            x2 = self.linear2(final_emb)
            x2 = gelu(x2)

        if not self.args.use_gumbel:
            x = torch.cat((x1, x2), -1)
            x1 = self.linear3(x1)
            weight = F.sigmoid(x1)

        elif is_loss:
            x1 = self.linear3(x1)
            weight = F.gumbel_softmax(F.log_softmax(x1, dim=-1), tau=tau, hard=True)
            weight = weight[:, 0]
        else:
            x = torch.cat((x1, x2), -1)

            x1 = self.linear3(x1)
            #pdb.set_trace()

            weight = F.gumbel_softmax(F.log_softmax(x1, dim=-1), tau=tau, hard=True)
            weight = weight[:, :, 0]

        if not is_loss:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
            if label_mask is not None:
                active_loss = active_loss & label_mask.view(-1)
            active_weight = weight.view(-1, 1)[active_loss]
        else:
            active_weight = weight

        #weight = nn.softmax(weight)
        #x = self.relu(x)
        #out = self.linear2(x)
        #pdb.set_trace()

        return active_weight