from typing import NamedTuple
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import argparse
import os
import json
import shutil


def torch2onnx(model,onnx_output_path="outputs/cortana_communication_enus_MV4.slots.rnn.onnx.bin"):
    
    dummy_input0 = torch.LongTensor(1, 128).to(torch.device("cuda"))
    dummy_input0.fill_(0)
    torch.onnx.export(model=model,
                        args=(dummy_input0,'onnx-export'), #change it to export mode
                        input_names = ["input_ids"], #Input mask is seperately handled in my training, it won't influence inference if we don't do dynamic padding
                        verbose=True,
                        output_names = ["output"],
                        do_constant_folding=True,
                        dynamic_axes = {'input_ids': {1: '?'}, 'output': {1: '?'}},
                        export_params=True,
                        f=onnx_output_path)

