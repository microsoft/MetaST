# Instructions to build : https://krishansubudhi.github.io/development/2019/09/23/CreatingDockerImage.html
# Existing images : krishansubudhi/transformers_pytorch:latest - built on 2_12_2020. change SHA below to pull latest apex
#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04
#FROM krishansubudhi/torch_azure:1.5

# NCCL 2.4 does not work with PyTorch, uninstall
#RUN apt-get update && apt-get --purge remove libnccl2 -y --allow-change-held-packages

#RUN apt-get -y update && apt-get -y install --no-install-recommends libnccl2=2.3.7-1+cuda10.0 libnccl-dev=2.3.7-1+cuda10.0


RUN ldconfig /usr/local/cuda/lib64/stubs && \
    # Install GPUtil
    pip install --no-cache-dir GPUtil && \
    # Install AzureML SDK
    pip install --no-cache-dir azureml-defaults && \
    # Install PyTorch
    #pip install --no-cache-dir https://download.pytorch.org/whl/cu100/torch-1.0.1-cp36-cp36m-linux_x86_64.whl &&\
    #pip install --no-cache-dir https://download.pytorch.org/whl/cu100/torch-1.3.0%2Bcu100-cp36-cp36m-linux_x86_64.whl &&\ 
    pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    #pip install --no-cache-dir torch==1.6.0+cu101  torchtext===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html && \


RUN pip install --no-cache-dir pytorch-pretrained-bert==0.6.2 && \
	pip install --no-cache-dir tensorboardX

RUN pip install --no-cache-dir transformers==2.10.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    tmux \
    unzip \
    htop

############################################
# Install Apex
# https://github.com/NVIDIA/apex/issues/988
############################################

RUN git clone https://github.com/NVIDIA/apex &&\
    cd apex &&\
    pip install  --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip install --no-cache-dir scipy && \
    pip install --no-cache-dir sklearn && \
    pip install --no-cache-dir matplotlib && \
    pip install --no-cache-dir seaborn && \
    pip install --no-cache-dir nltk && \
    pip install --no-cache-dir tensorboard && \
    pip install --no-cache-dir sentencepiece && \
    pip install --no-cache-dir msgpack &&\
    pip install --no-cache-dir psutil &&\
    pip install --no-cache-dir rouge &&\
    pip install --no-cache-dir scikit-learn &&\
    pip install --no-cache-dir seqeval &&\
    pip install --no-cache-dir flashtool &&\
    pip install --no-cache-dir wandb &&\
    pip install --no-cache-dir pandas
    

RUN python -c "import nltk; nltk.download('punkt')"
RUN pip install -I https://matrixdeploystorageeus.blob.core.windows.net/matrixlib/matrixlib-0.1.2018481.tar.gz
RUN pip install higher


RUN conda install pip=20.1.1
RUN pip install --upgrade azureml-sdk[notebooks,contrib]
RUN pip install cryptography==3.4.1