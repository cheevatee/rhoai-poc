import boto3
import os

def upload_folder_to_s3(local_dir, bucket_name, s3_folder):
    s3 = boto3.client('s3')
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.join(s3_folder, os.path.relpath(local_path, local_dir))
            s3.upload_file(local_path, bucket_name, s3_path)

upload_folder_to_s3("./qwen3-final", "my-model-bucket", "exports/qwen3-mars/")
