import boto3
import json
import uuid
import os
from datetime import datetime

# def submit_nma_batch_job(user, model_id, dataset, graph_type, min_confidence, top_k):
def submit_nma_batch_job(user_id):
    batch = boto3.client('batch', region_name=os.getenv("AWS_REGION"),
                        aws_access_key_id = os.getenv("AWS_JOBS_ACCESS_KEY_ID"),
                        aws_secret_access_key = os.getenv("AWS_JOBS_SECRET_ACCESS_KEY")) 

    job_name = f"nma-job-{uuid.uuid4()}"
    job_queue = os.getenv("JOB_QUEUE_NAME")
    job_definition = os.getenv("JOB_DEFINITION_NAME")

    # parameters = {
    #     "user_id": user.id,
    #     "model_id": model_id,
    #     "dataset": dataset,
    #     "graph_type": graph_type,
    #     "min_confidence": str(min_confidence),
    #     "top_k": str(top_k),
    #     "timestamp": datetime.now(datetime.timezone.utc)
    # }
    
    parameters = {
        "user_id": user_id,
    }

    response = batch.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
    )

    if response.get("jobId"):
        return response["jobId"]
    else:
        raise Exception(f"Failed to submit batch job: {response}")
