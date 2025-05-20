# test_s3_structure.py
import os
import boto3
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv(r"C:\Users\adler\BETTER\.env")

# Create S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)

bucket = os.environ.get('S3_BUCKET_NAME')
print(f"Connected to bucket: {bucket}")

# Test downloading from cifar100 folder
temp_dir = tempfile.mkdtemp()
try:
    print(f"\nTesting download from cifar100/ folder...")
    
    # List files in the cifar100 folder
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix="cifar100/"
    )
    
    if 'Contents' in response and response['Contents']:
        print(f"Found {len(response['Contents'])} files in cifar100/ folder")
        print("First 5 files:")
        for i, obj in enumerate(response['Contents'][:5]):
            print(f"- {obj['Key']}")
        
        # Download one file as a test
        if response['Contents']:
            test_file = response['Contents'][0]['Key']
            filename = os.path.basename(test_file)
            local_path = os.path.join(temp_dir, filename)
            
            print(f"\nDownloading {test_file} to {local_path}...")
            s3.download_file(bucket, test_file, local_path)
            
            if os.path.exists(local_path):
                print(f"✅ Successfully downloaded! File size: {os.path.getsize(local_path)} bytes")
            else:
                print("❌ Download seems to have failed - file not found")
    else:
        print("No files found in cifar100/ folder")
    
    # Also test downloading cifar100_info.py
    info_file = "cifar100_info.py"
    info_path = os.path.join(temp_dir, info_file)
    
    print(f"\nDownloading {info_file}...")
    try:
        s3.download_file(bucket, info_file, info_path)
        print(f"✅ Successfully downloaded info file! Size: {os.path.getsize(info_path)} bytes")
    except Exception as e:
        print(f"❌ Failed to download info file: {str(e)}")
    
finally:
    # Clean up
    shutil.rmtree(temp_dir)
    print("\nCleaned up temporary directory")