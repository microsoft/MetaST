
# Requirements
To run this script, you only need to activate amlsdk locally with virtual environment. The remaining enviornment will be configurated in Aml.





# Train and Evaluate in aml

To create a specific experiment, you need to connect to aml cluster and choose corresponding cluster. For instance, the following command will connect to the default aml workspace and use 'p100cluster' for 'debug' experiment

`python teacher_supervised_finetune_aml.py --cluster_name p100cluster --experiment_name debug`

if you want to connect to different workspace, download the config.json in the workspace and put it in the root dir:

`python teacher_supervised_finetune_aml.py --cluster_name p100cluster --experiment_name debug --config_dir .`

# Distillation

`python kd_aml.py --cluster_name p100cluster --experiment_name debug`

# Inference

```python
from bert_inference import Ner

model = Ner("out/")

output = model.predict("Steve went to Paris")

```

## Data

Currently, all of my training data and teacher model is stored in azureml blob. In the folder of dataset. You can find the link here  

https://ms.portal.azure.com/#blade/Microsoft_Azure_Storage/ContainerMenuBlade/overview/storageAccountId/%2Fsubscriptions%2F4a66f470-dd54-4c5e-bd19-8cb65a426003%2FresourceGroups%2FAML_Playground%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fteamsws4000114466/path/azureml-blobstore-1dd9ddd7-6d52-40ad-8b00-be73cd90fd63/etag/%220x8D6D3FDFD25EA50%22

Aml provide one easy-to-use plugin https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-azurestorage to visulaize the blob or find the data in code.


``` python
# List all datastores registered in the current workspace
datastores = ws.datastores
for name, datastore in datastores.items():
    print(name, datastore.datastore_type)
```

I also registered this blob to MSAI subscription, which means you can submit job in that subscription with one simple change in code:

``` python
from azureml.core import Workspace,Datastore 
ds = Datastore.get(ws, datastore_name='compliant_lu_haochu')
```

## Data format

For labeled data, I use conll format. For unlabeled data, it is saved line by line. Please find sample data from sample_data folder.


# Reproduce the results of teacher model
To reproduce the, please use the comm_train_prod.txt in teacher_supervised_finetune_aml.py and this will generate the teacher model 

# Noteboook
One may also choose to submit the job in jupyter notebook
