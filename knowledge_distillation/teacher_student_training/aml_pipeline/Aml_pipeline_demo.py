
import azureml.core
from azureml.core import Workspace,Datastore,Experiment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import PipelineData,Pipeline
from azureml.pipeline.steps import PythonScriptStep,EstimatorStep

from azureml.train.dnn import PyTorch

class Teacher_Training_Step(EstimatorStep):

    def __init__(self,
                 ds,
                 data_dir,
                 train_dir,
                 valid_dir,
                 test_generated_dir,
                 test_generated_no_contact_dir,
                 target_set_dir,
                 output_dir: PipelineData,
                 compute_target):

        pip_packages=['pandas','pytorch-pretrained-bert==0.6.1','seqeval==0.0.5','transformers==2.1.1']
        entry_script = "src/Teacher_training.py"
        self.estimator = PyTorch(
            source_directory='.',
            compute_target=compute_target,
            entry_script=entry_script,
            pip_packages=pip_packages,
            use_gpu=True
        )
        
        
        args = [
            '--data_dir', data_dir,
            '--train_dir',train_dir,
            '--valid_dir', valid_dir,
            '--test_generated_dir', test_generated_dir,
            '--test_generated_no_contact_dir', test_generated_no_contact_dir,
            '--target_set_dir', target_set_dir,
            

            '--bert_model', 'bert-base-uncased',
            '--do_lower_case', 
            #'--do_basic_tokenize', 
            
            '--task_name', 'ner',
            
            '--output_dir', output_dir,
            '--do_train', 
            '--do_eval', 
            '--learning_rate', 2e-5,
            '--num_train_epochs', 20,
            '--warmup_proportion', 0.1,
            '--max_seq_length', 128,
            '--train_batch_size', 64
        ]
       # print (args)
        super().__init__(
            name="Build Teacher training step",
            estimator=self.estimator,
            estimator_entry_script_arguments=args,
            inputs=[data_dir,train_dir,valid_dir,test_generated_dir,test_generated_no_contact_dir,target_set_dir], 
            outputs=[output_dir],
            compute_target=compute_target,
            allow_reuse=True
        )



class Student_Training_Step(EstimatorStep):

    def __init__(self,
                 ds,
                 data_dir,
                 train_dir,
                 valid_dir,
                 test_generated_dir,
                 test_generated_no_contact_dir,
                 target_set_dir,
                 unsupervised_train_corpus,
                 teacher_model_path: PipelineData,
                 output_dir: PipelineData,
                 compute_target):

        pip_packages=['pandas','pytorch-pretrained-bert==0.6.1','seqeval==0.0.5','transformers==2.1.1','nltk']
        entry_script = "src/distillation.py"
        self.estimator = PyTorch(
            source_directory='.',
            compute_target=compute_target,
            entry_script=entry_script,
            pip_packages=pip_packages,
            use_gpu=True
        )

        
        
        args = [
            '--data_dir', data_dir,
            '--train_dir',train_dir,
            '--valid_dir', valid_dir,
            '--test_generated_dir', test_generated_dir,
            '--test_generated_no_contact_dir', test_generated_no_contact_dir,
            '--target_set_dir', target_set_dir,
            '--teacher_model_path', teacher_model_path,
            '--unsupervised_train_corpus', unsupervised_train_corpus,

            '--bert_model', 'bert-base-uncased',
            '--do_lower_case', 
            #'--do_basic_tokenize', 
            
            '--task_name', 'ner',
            
            '--output_dir', output_dir,
            '--do_train', 
            '--do_eval', 
            '--learning_rate', 1e-4,
            '--num_train_epochs', 50,
            '--warmup_proportion', 0.1,
            '--max_seq_length', 128,
            '--train_batch_size', 64,
            '--alpha', 1,
            '--beta', 1,
            '--encoder_type', 'GRU',
            '--hidden_units',300

        ]
        

       # print (args)
        super().__init__(
            name="Build Student model",
            estimator=self.estimator,
            estimator_entry_script_arguments=args,
            inputs=[data_dir,train_dir,valid_dir,test_generated_dir,test_generated_no_contact_dir,target_set_dir,teacher_model_path,unsupervised_train_corpus], 
            outputs=[output_dir],
            compute_target=compute_target,
            allow_reuse=True
        )





    



if __name__=='__main__':
    #subscription_id = "4a66f470-dd54-4c5e-bd19-8cb65a426003"
    #resource_group  = "AML_Playground"
    #workspace_name  = "Teams_ws"

    subscription_id = "ddb33dc4-889c-4fa1-90ce-482d793d6480"
    resource_group = "DevExp"
    workspace_name = "DevExperimentation"

    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)


    #ws = Workspace.from_config() 
    #blob_store = ws.get_default_datastore()
    blob_store = Datastore(ws, "compliant_lu_haochu")
    #blob_store = Datastore(ws, "workspaceblobstore")

    compute_target = ws.compute_targets["P100-SingleGPU"]
    experiment = Experiment(ws, 'Communication_Lu') 


    teacher_model_output_path = PipelineData(
        "demo_teacher",
        datastore=blob_store,
        #output_path_on_compute = 'pipeline_output/demo_teacher',
        #output_mode='mount',
        #is_directory = True,
        output_name="pipeline_output_teacher_model")


    student_model_output_path = PipelineData(
        "demo_student",
        datastore=blob_store,
        #output_path_on_compute = 'pipeline_output/demo_student',
        #output_mode='mount',
        #is_directory = True,
        output_name="pipeline_output_student_model")


    
    data_dir = DataReference(
            datastore=blob_store,
            data_reference_name="data_dir",
            path_on_datastore=f'datasets/Communication_prod_data')

    train_dir = DataReference(
            datastore=blob_store,
            data_reference_name="train_dir",
            path_on_datastore=f'datasets/Communication_prod_data/communication_slot_train.txt')
            #path_on_datastore=f'datasets/Communication_prod_data/communication_slot_train_3_15.txt')
        
    valid_dir = DataReference(
            datastore=blob_store,
            data_reference_name="valid_dir",
            path_on_datastore=f'datasets/Teams_communication/valid_dec.txt')

    test_generated_dir = DataReference(
            datastore=blob_store,
            data_reference_name="test_generated_dir",
            path_on_datastore=f'datasets/Teams_communication/generated_data/communication_message_generated_no_contact.txt')
        
    test_generated_no_contact_dir = DataReference(
            datastore=blob_store,
            data_reference_name="test_generated_no_contact_dir",
            path_on_datastore=f'datasets/Teams_communication/generated_data/communication_message_generated_contact.txt')

    target_set_dir = DataReference(
            datastore=blob_store,
            data_reference_name="target_set_dir",
            path_on_datastore=f'datasets/Teams_communication/Target_set_message_new_conll.txt')



    unsupervised_train_corpus = DataReference(
            datastore=blob_store,
            data_reference_name="unsupervised_train_corpus",
            path_on_datastore=f'datasets/Teams_communication/raw_generated_data/raequery_for_prod_with_target.txt')



    


    


    Teacher_Training = Teacher_Training_Step(blob_store,data_dir, train_dir, valid_dir, test_generated_dir,
                                            test_generated_no_contact_dir, target_set_dir, teacher_model_output_path,compute_target)
    Student_training = Student_Training_Step(blob_store,data_dir, train_dir, valid_dir, test_generated_dir,
                                            test_generated_no_contact_dir, target_set_dir, unsupervised_train_corpus, teacher_model_output_path, student_model_output_path,compute_target)

    steps = [Teacher_Training,Student_training]

    pipeline = Pipeline(workspace=ws, steps=steps)

    pipeline_run = experiment.submit(pipeline)
    pipeline_run.wait_for_completion()


    
    #from azureml.widgets import RunDetails
    #RunDetails(pipeline_run).show()