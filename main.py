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
import requests


PROJECT_ID = "ameai-causal"
REGION = "asia-east1"
MODEL_BUCKET = "causal_models"
MODEL_BUCKET_URI = "gs://causal_models"
STAGING_BUCKET_URI = "gs://causal_data"

url = 'http://34.81.134.109:80/inference'

@functions_framework.http
def inference(request):
    user_input = request.get_json()

    response = requests.post(
        url,
        json=user_input
    )

    return response
    