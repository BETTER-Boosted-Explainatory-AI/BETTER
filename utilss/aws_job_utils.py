import boto3
import json
import uuid
import os
from datetime import datetime

# TODO: change it all to get these parameters: (user_id, model_id, dataset, graph_type, min_confidence, top_k)

def submit_nma_batch_job(user_id, model_id, dataset, graph_type, min_confidence, top_k):
    batch = boto3.client('batch', region_name=os.getenv("AWS_REGION"),
                        aws_access_key_id = os.getenv("AWS_JOBS_ACCESS_KEY_ID"),
                        aws_secret_access_key = os.getenv("AWS_JOBS_SECRET_ACCESS_KEY")) 

    job_name = f"nma-job-{uuid.uuid4()}"
    job_queue = os.getenv("JOB_QUEUE_NAME")
    job_definition = os.getenv("JOB_DEFINITION_NAME")
    
    environment = [
        {'name': 'user_id', 'value': str(user_id)},
        {'name': 'model_id', 'value': str(model_id)},
        {'name': 'dataset', 'value': str(dataset)},
        {'name': 'graph_type', 'value': str(graph_type)},
        {'name': 'min_confidence', 'value': str(min_confidence)},
        {'name': 'top_k', 'value': str(top_k)},
    ]

    response = batch.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides={
            'environment': environment
        }
    )

    if response.get("jobId"):
        return response["jobId"]
    else:
        raise Exception(f"Failed to submit batch job: {response}")
