"""Step 5 (run locally): deploy the trained model to a real-time SageMaker endpoint.

The model takes raw JPEG bytes (decode and resize are inside the SavedModel), so no
inference script is needed: TF Serving runs the model as saved.
The endpoint is billed per hour while it exists: delete it right after evaluation.

Run: python 05_deploy.py
"""
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

from config import REGION, EXECUTION_ROLE_ARN, S3_OUTPUT_PREFIX

TRAINING_JOB = "cats-vs-dogs-vgg16-2026-11-21-15-38-36-553"   # tuned best model, JPEG serving signature
ENDPOINT_NAME = "cats-vs-dogs-endpoint"

sess = sagemaker.Session(boto3.Session(region_name=REGION))
model_data = f"s3://{sess.default_bucket()}/{S3_OUTPUT_PREFIX}/{TRAINING_JOB}/output/model.tar.gz"

model = TensorFlowModel(
    model_data=model_data,
    role=EXECUTION_ROLE_ARN,
    framework_version="2.14",        # same TF version as training
    sagemaker_session=sess,
)

model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",    # CPU; no GPU endpoint quota needed
    endpoint_name=ENDPOINT_NAME,     # fixed name so the evaluation script can find and delete it
)

print("Endpoint in service:", ENDPOINT_NAME)
