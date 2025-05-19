from dotenv import load_dotenv
import os, boto3, pathlib

load_dotenv()                                           # skips silently if no .env

bucket = os.getenv("S3_BUCKET_NAME", "better-datasets")
dataset_key = "cifar100/"                               # folder prefix in S3
local_root  = pathlib.Path("data/cifar100")             # where to save files
aws_region = os.getenv("AWS_REGION")
s3 = boto3.client("s3", aws_region)                                 # keys auto-loaded

def download_prefix(bucket, prefix, destination):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel_path = pathlib.Path(obj["Key"]).relative_to(prefix)
            local_path = destination / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"↓ {obj['Key']} → {local_path}")
            s3.download_file(bucket, obj["Key"], str(local_path))

download_prefix(bucket, dataset_key, local_root)
