from utilss.s3_utils import get_users_s3_client 

# #!/usr/bin/env python
# # test_load_model.py

# from dotenv import load_dotenv
# load_dotenv()   # loads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION…

# import os
# import boto3
# import tensorflow as tf

# # 1. S3 URI of your .keras file
# s3_uri = (
#     "s3://better-xai-users/"
#     "92457494-3061-700c-8e14-f1ab249392d7/"
#     "33b8e5d1-c7d2-4c35-8a29-5764e8dd86fd/"
#     "cifar100_resnet.keras"
# )

# # 2. Quick boto3 head_object check
# bucket = "better-xai-users"
# key = (
#     "92457494-3061-700c-8e14-f1ab249392d7/"
#     "33b8e5d1-c7d2-4c35-8a29-5764e8dd86fd/"
#     "cifar100_resnet.keras"
# )
# s3 = boto3.client("s3")
# try:
#     s3.head_object(Bucket=bucket, Key=key)
#     print("✅ S3 object exists")
# except Exception as e:
#     print("❌ S3 head_object failed:", e)
#     exit(1)

# # 3. Check TensorFlow’s file API sees it
# if tf.io.gfile.exists(s3_uri):
#     print("✅ tf.io.gfile.exists sees the file")
# else:
#     print("❌ tf.io.gfile.exists DOES NOT see the file — check your AWS creds / region")
#     exit(1)

# # # 4. Load the model and print its architecture
# # try:
# #     model = tf.keras.models.load_model(s3_uri)
# #     print("\n✅ Model loaded! Summary:\n")
# #     model.summary()
# # except Exception as e:
# #     print("❌ Failed to load model with tf.keras.models.load_model:", e)
# #     exit(1)

# # # 5. (Optional) Use your own helper to load it
# # from services.models_service import _load_model, _get_dataset_config
# # dataset_str = "cifar100"
# # cfg = _get_dataset_config(dataset_str)
# # try:
# #     wrapped_model = _load_model(dataset_str, key, cfg)
# #     print("\n✅ _load_model helper succeeded. Wrapped model summary:\n")
# #     wrapped_model.model.summary()
# # except Exception as e:
# #     print("❌ _load_model helper failed:", e)
# #     exit(1)

from dotenv import load_dotenv
load_dotenv()

import boto3
import logging

# Enable detailed AWS logging
logging.basicConfig(level=logging.INFO)
boto3_logger = logging.getLogger('boto3')
boto3_logger.setLevel(logging.DEBUG)
botocore_logger = logging.getLogger('botocore')
botocore_logger.setLevel(logging.DEBUG)

# Print which identity is being used
sts = boto3.client('sts')
try:
    caller_identity = sts.get_caller_identity()
    print(f"AWS Identity: {caller_identity['Arn']}")
except Exception as e:
    print(f"Error getting identity: {e}")

# Explicitly create the client with the correct profile
s3_client = get_users_s3_client()

