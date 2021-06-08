import azureml.core
from azureml.core import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core import Experiment
#from azureml.widgets import RunDetails
from azureml.train.dnn import PyTorch
from aml_utils.aml_helper import *
import argparse



def create_estimator(ws, compute_target, ds, args):

    script_params = {
    
    '--data_dir': ds.path(f'datasets/Teams_communication').as_mount(), #update for communication data
    
    '--train_dir':ds.path(f'datasets/Teams_communication/comm_train_prod.txt').as_mount(),
    '--eval_dir':ds.path(f'datasets/Teams_communication/generated_data/communication_message_generated_no_contact.txt').as_mount(),
    '--test_dir':ds.path(f'datasets/Teams_communication/generated_data/communication_message_generated_contact.txt').as_mount(),
    '--target_set_dir':ds.path(f'datasets/Teams_communication/generated_data/Target_set_message_new_conll.txt').as_mount(),
    
    
    ##for uncased longer teacher
    '--teacher_model_path':ds.path(f'bert_data/uncased_model/outputs_base_uncased_no_basic_tokenizer').as_mount(),#from prod labeled sources
    '--bert_model':'bert-base-uncased', 
    '--do_lower_case':'',
    
    
    '--task_name':'ner',
    '--output_dir':'./outputs',
    '--do_train':'',
    '--do_eval':'',
    #'--crf':'',
    '--learning_rate':'1e-4',#smaller learning rate given init
    '--num_train_epochs':'50', ##get larger number when you have smaller unsupervised data
    '--warmup_proportion':'0.1',
    #'--max_seq_length':'32',
    '--max_seq_length':'128',
    '--train_batch_size':'64',
    "--temperature": '1',
    "--alpha": '1',#for ablation study,weight for labeled data
    "--beta": '1', #for ablation study,weight for unlabeled data
    #"--unsupervised_train_corpus":'./lm_data/train.txt' #add for debug
    #"--unsupervised_train_corpus":'./lm_data/augmented_data.txt' #add for debug
    #'--unsupervised_train_corpus':ds.path(f'datasets/Teams_communication/raw_generated_data/rawquery_merged.txt').as_mount()
    #'--unsupervised_train_corpus':ds.path(f'datasets/Teams_communication/raw_generated_data/rawquery_for_prod.txt').as_mount()
    #'--unsupervised_train_corpus':ds.path(f'datasets/Teams_communication/raw_generated_data/rawquery_for_prod_from_valid_nov_normalized.txt').as_mount()   
    #'--unsupervised_train_corpus':ds.path(f'datasets/Teams_communication/raw_generated_data/rawquery_from_civ.txt').as_mount()
    #'--unsupervised_train_corpus':ds.path(f'datasets/Teams_communication/raw_generated_data/raequery_for_prod_with_target.txt').as_mount()
    '--unsupervised_train_corpus':ds.path(f'datasets/Teams_communication/raw_generated_data/rawquery_for_prod_with_civ.txt').as_mount()
    

    #'--unsupervised_train_corpus':ds.path(f'datasets/Teams_communication/raw_generated_data/rawquery_expanision-outputs.txt').as_mount()
    }



    pytorch_estimator = PyTorch(source_directory='.', 
                        script_params=script_params,
                        compute_target=compute_target, 
                        entry_script='src/distillation.py',
                        #pip_packages=['pandas','pytorch-pretrained-bert==0.4.0','seqeval==0.0.5'],
                        pip_packages=['pandas','pytorch-pretrained-bert==0.6.1','seqeval==0.0.5','transformers==2.1.1','nltk'],
                        use_gpu=True)

    print(pytorch_estimator.run_config.environment.docker.base_image)
    print(pytorch_estimator.conda_dependencies.serialize_to_string())

    return pytorch_estimator

def experiment_config(parser):
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The input data dir. Should be conll format")
    parser.add_argument("--config_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="aml config")
    parser.add_argument("--experiment_name",
                        default='debug',
                        type=str,
                        required=True,
                        help="The experiment name of aml job")
    parser.add_argument("--cluster_name",
                        default='p100cluster',
                        type=str,
                        required=True,
                        help="The cluster name of aml job")
    parser.add_argument("--output_dir",
                        default='./outputs',
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    


    return parser.parse_args()


if __name__=='__main__':
    

    parser = argparse.ArgumentParser()
    args = experiment_config(parser)
    print("SDK version:", azureml.core.VERSION)

    
    if args.config_dir:
        # load workspace configuration from the config.json file in the current folder.
        ws = Workspace.from_config()
    else:
        ws = connect_to_workspace(subscription_id = "4a66f470-dd54-4c5e-bd19-8cb65a426003",
                                resource_group  = "AML_Playground",
                                workspace_name  = "Teams_ws")
    compute_target = connect_to_cluster(ws, cluster_name = args.cluster_name)

    ##get access to the datastore
    ds = ws.get_default_datastore()

    
    experiment = Experiment(workspace = ws, name=args.experiment_name)


    pytorch_estimator = create_estimator(ws,compute_target,ds, args)

    run = experiment.submit(pytorch_estimator)
    