import azureml.core
from azureml.core import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core import Experiment
from azureml.train.dnn import PyTorch





def connect_to_workspace(subscription_id,resource_group,workspace_name):




    try:
        ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
        ws.write_config()
        print('Library configuration succeeded')
        print('https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource' + ws.get_details()['id'])
    except:
        print('Workspace not found')
    
    return ws



def connect_to_cluster(ws, cluster_name = "p100cluster"):
    try:
        compute_target = ws.compute_targets[cluster_name]
        print('Found existing compute target.')
    except KeyError:
        print('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6s_v2', 
                                                            idle_seconds_before_scaledown=1800,
                                                            min_nodes=0, 
                                                            max_nodes=10)
        # create the cluster
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True)
    return compute_target
