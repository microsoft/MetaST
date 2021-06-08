
# Files

* data_utils.py: read and process dataset

* eval.py: do evaluation 

* run_self_training_active_learning.py: run self-training with weighted sampling on labeled data

* run_self_training_teacher_finetune.py:  run self-training

* sampler.py: implement four sampling methods(variance of loss, decay of loss, variance of prediciton, uniform)

* modelining_bert: bert model file

* modeling_meta_bert: used in the meta-learning, adapat the parameter and conduct inference with adapated parameters.

* modeling_utils: aggregate psuedo labels, a main component in the BOND. We did not have specific design here.






## Explorations 

* meta_controler.py: used for the meta-weight net option. This is also not in our final framework

* modeling_roberta.py roberta file but not in our final framework

* run_self_training_self_paced.py:  run self-training with self-paced learning [Easy sample first]

* run_self_training_teacher_augmentation.py:  run self-training with extra augmentation loss on labeled data

* focal_loss.py: implement focal loss which is used in the hard-data mining [Larger weight on Hard samples]




# Data format

For unlabeled and labeled data, I use conll format. 
