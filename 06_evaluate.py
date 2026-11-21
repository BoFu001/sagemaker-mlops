"""Step 6 (run locally): evaluate the endpoint on the held-out test set, then delete it.

Reads the test JPEGs straight from S3, sends them to the endpoint in small batches,
and reports accuracy, precision, recall, F1 and the confusion matrix (dog = positive).
The endpoint, its config and the model are deleted at the end, even if evaluation fails.

Run: python 06_evaluate.py
"""
import base64
import json

import boto3
import numpy as np
import sagemaker
from sagemaker.predictor import Predictor

from config import REGION, S3_DATA_PREFIX, CLASS_NAMES

ENDPOINT_NAME = "cats-vs-dogs-endpoint"
BATCH = 32                                   # ~1 MB per request, well under the 6 MB payload limit

sess = sagemaker.Session(boto3.Session(region_name=REGION))
bucket = sess.default_bucket()                      # sagemaker-eu-central-1-766577086453
s3 = sess.boto_session.client("s3")                 # read the test images
runtime = sess.boto_session.client("sagemaker-runtime")   # call the endpoint

# Test set: (S3 key, label) with cats=0, dogs=1, in a fixed order
items = []
for label, cls in enumerate(CLASS_NAMES):
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{S3_DATA_PREFIX}/test/{cls}/")
    items += sorted((obj["Key"], label) for page in pages for obj in page.get("Contents", []))
print("Test images:", len(items))

try:
    probs = []
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        instances = [{"b64": base64.b64encode(s3.get_object(Bucket=bucket, Key=k)["Body"].read()).decode()}
                     for k, _ in batch]
        resp = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType="application/json",
                                       Body=json.dumps({"instances": instances}))
        probs += np.ravel(json.loads(resp["Body"].read())["predictions"]).tolist()
        print(f"{len(probs)}/{len(items)}", end="\r")

    y_true = np.array([label for _, label in items])
    y_pred = (np.array(probs) > 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision, recall = tp / (tp + fp), tp / (tp + fn)

    print(f"\nAccuracy:  {(tp + tn) / len(items):.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-score:  {2 * precision * recall / (precision + recall):.2%}")
    print(f"True cats predicted as cats: {tn}")
    print(f"True cats predicted as dogs (FP): {fp}")
    print(f"True dogs predicted as cats (FN): {fn}")
    print(f"True dogs predicted as dogs: {tp}")
finally:
    predictor = Predictor(ENDPOINT_NAME, sagemaker_session=sess)
    predictor.delete_model()                 # needs the endpoint config, so delete the model first
    predictor.delete_endpoint()              # also deletes the endpoint config
    print("Endpoint, endpoint config and model deleted.")
