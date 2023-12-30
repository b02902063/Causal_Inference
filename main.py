import functions_framework
import os
import uuid
from datetime import datetime
import json
from google.cloud import aiplatform
import torch
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from tensordict import TensorDict
from google.cloud import storage


PROJECT_ID = "ameai-causal"
REGION = "asia-east1"
MODEL_BUCKET_URI = "gs://causal_models"
STAGING_BUCKET_URI = "gs://causal_data"

storage_client = storage.Client()

def get_model_from_gcs(bucket, name):
    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(name)
    with blob.open("rb") as f:
        model = torch.load(f)   
    return model
    

@functions_framework.http
def inference(request):
    user_input = request.get_json()
    
    model_id = user_input["model_id"]
    
    loaded_model = get_model_from_gcs(MODEL_BUCKET_URI, model_id + ".pt")
    #torch.load(os.path.join(MODEL_BUCKET_URI, model_id) + ".pt")
    
    normalizer = loaded_model["normalizer"]
    graph = loaded_model["new_adg"]
    model = loaded_model["model"]
    
    SEM_MODULE = lightning_module.sem_module()
    sem = DistributionParametersSEM(graph, SEM_MODULE._noise_module, SEM_MODULE._functional_relationships)
    
    before = dict()
    after = dict()
    for key, value in user_input["before"].items():
        before[key] = torch.tensor([value]).float()

    for key, value in user_input["after"].items():
        after[key] = torch.tensor([value]).float()
        
    before = TensorDict(before, batch_size=tuple())
    after = TensorDict(after, batch_size=tuple())
    
    intervention_a = normalizer(before)
    intervention_b = normalizer(after)
    
    torch_shape = torch.Size([20000])
    
    rev_a_samples = normalizer.inv(sem.do(interventions=intervention_a).sample(torch_shape))
    rev_b_samples = normalizer.inv(sem.do(interventions=intervention_b).sample(torch_shape))
    
    output = {}
    for key in rev_a_samples.keys():
        ate_mean = (rev_b_samples[key].mean(0) - rev_a_samples[key].mean(0)).detach().cpu().numpy()[0].item()
        output[key] = ate_mean
    
    return output
    