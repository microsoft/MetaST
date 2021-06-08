import numpy as np
import time, os, math, operator, statistics, sys
from random import Random
import torch
import torch.utils.data
import torchvision
import pdb
import torch.nn.functional as F
class UncertaintyDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """
    def __init__(self, dataset, num_samples=None, smoothness_type='mean', mode='loss_decay'):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.smoothness_type = smoothness_type
        self.indices = {}
        self.all_values = {}
        self.mode = mode
        self.smoothness = 0

        for i, f in enumerate(dataset):
            uid = f[-1].item()
            self.indices[uid] = i
            self.all_values[uid] = []



        # # define custom callback
        # self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration

        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        self.weights = [10.0] * self.num_samples






    def _compute_sample_weights(self, ids, latest_n=5):

        for i in range(len(ids)):
            id = ids[i]

            values = self.all_values[id]
            if len(values) <= 1:
                weight = 10.0
            elif self.mode == 'loss_decay':
                weight = max(np.mean(self.all_values[id][-latest_n:-1]) - self.all_values[id][-1], 0)
                # if  self.smoothness is not None:
                #     weight = weight + self.smoothness
                self.all_values[id] = self.all_values[id][-latest_n:]
            elif self.mode == 'loss_var':
                variance = np.var(values)
                weight = np.sqrt(variance)
                # if self.smoothness is not None:
                #     weight = weight + self.smoothness
                self.all_values[id] = self.all_values[id][-latest_n:]
            else:
                variance = np.var(values)
                weight = variance + (variance * variance) / (float(len(values)) - 1.0)
                weight = np.sqrt(weight)
                # if  self.smoothness is not None:
                #     weight = weight + self.smoothness
                self.all_values[id] = self.all_values[id][-latest_n:]

            self.weights[self.indices[id]] = weight
        if self.smoothness_type == 'mean':
            self.smoothness = np.mean(self.weights)
        else:
            self.smoothness = np.max(self.weights)



    def async_update_matrix(self, ids, labels, outputs):
        #pdb.set_trace()



        if 'loss' in self.mode:
            outputs = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), reduce=False).view(labels.size(0), labels.size(1))
        elif self.mode == 'logits':
            outputs = F.softmax(outputs)

        outputs = outputs.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        ids = ids.cpu().numpy()


        for i in range(len(ids)):
            id = ids[i]
            temp_values = []
            for j, label in enumerate(labels[i]):
                if label >= 0 :
                    if 'loss' in self.mode:
                        loss = outputs[i][j]
                        temp_values.append(loss)

                    else:
                        probability = outputs[i][j][label]
                        temp_values.append(probability)

            mean_value = np.mean(temp_values)

            # append the prediction probability to the map
            self.all_values[id].append(mean_value)

        self._compute_sample_weights(ids)




    def __iter__(self):
        sampling_weights = self.smoothness + np.array(self.weights) + 0.00001

        return iter(torch.multinomial(torch.tensor(sampling_weights).float(), self.num_samples, replacement=False).tolist())


    def __len__(self):
        return self.num_samples