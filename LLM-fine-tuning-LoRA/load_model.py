import boto3
import os

def download_s3_folder(bucket_name, s3_folder, local_dir):
    s3 = boto3.client('s3')
    # Assumes environment variables for AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in the workbench
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)['Contents']
    for obj in objects:
        local_file_path = os.path.join(local_dir, os.path.relpath(obj['Key'], s3_folder))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        s3.download_file(bucket_name, obj['Key'], local_file_path)

# download_s3_folder('my-model-bucket', 'models/qwen3-0.6b-base/', './qwen3-base')
