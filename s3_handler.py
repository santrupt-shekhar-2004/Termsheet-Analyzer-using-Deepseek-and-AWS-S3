import os
import json
import boto3
from botocore.exceptions import ClientError, ParamValidationError
import uuid
import logging

class S3Handler:
    def __init__(self, bucket_name=None, region_name=None):
        """
        Initialize S3 handler with bucket name from environment or parameter
        
        Priority for bucket name:
        1. Passed bucket_name parameter
        2. S3_BUCKET_NAME environment variable
        3. Generated unique bucket name
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Priority 1: Parameter
        if bucket_name:
            self.bucket_name = bucket_name
        # Priority 2: Environment Variable
        else:
            self.bucket_name = os.environ.get('S3_BUCKET_NAME')
        
        # Priority 3: Generate unique bucket name if still not set
        if not self.bucket_name:
            self.bucket_name = f"derivative-termsheets-{uuid.uuid4().hex[:8]}"
        
        # Use region from environment or parameter, default to us-east-1
        self.region_name = region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        
        # Create S3 client and resource
        self.s3_client = boto3.client('s3', region_name=self.region_name)
        self.s3_resource = boto3.resource('s3', region_name=self.region_name)
        
        # Create bucket if it doesn't exist
        self._create_bucket_if_not_exists()
        
        # Log bucket information
        self.logger.info(f"S3 Bucket Initialized: {self.bucket_name} (Region: {self.region_name})")

    def _create_bucket_if_not_exists(self):
        """
        Create S3 bucket if it doesn't already exist
        """
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.info(f"Bucket {self.bucket_name} already exists.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            # If bucket does not exist (404), create it
            if error_code == '404':
                try:
                    # Create bucket
                    create_bucket_params = {
                        'Bucket': self.bucket_name,
                        'CreateBucketConfiguration': {'LocationConstraint': self.region_name}
                    }
                    
                    # Only add LocationConstraint if not us-east-1
                    if self.region_name != 'us-east-1':
                        self.s3_client.create_bucket(**create_bucket_params)
                    else:
                        # us-east-1 doesn't use LocationConstraint
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    
                    self.logger.info(f"Created new S3 bucket: {self.bucket_name}")
                except ParamValidationError:
                    # Fallback for us-east-1
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                except Exception as create_error:
                    self.logger.error(f"Error creating bucket: {create_error}")
                    raise
            elif error_code == '403':
                self.logger.error(f"Access denied to bucket {self.bucket_name}")
                raise
            else:
                # Some other error occurred
                self.logger.error(f"Unexpected error checking bucket: {e}")
                raise

    def upload_file(self, file_path: str, s3_key: str) -> bool:
        """
        Upload a file to S3 with enhanced error handling and logging
        """
        try:
            # Ensure the key does not start with a /
            s3_key = s3_key.lstrip('/')
            
            # Determine content type
            content_type = self._get_content_type(file_path)
            
            # Upload file
            self.s3_client.upload_file(
                file_path, 
                self.bucket_name, 
                s3_key,
                ExtraArgs={'ContentType': content_type}
            )
            self.logger.info(f"Successfully uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            self.logger.error(f"Error uploading file to S3: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_content_type(self, file_path):
        """
        Determine content type based on file extension
        """
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'

    def upload_data(self, data, s3_key: str) -> bool:
        """
        Upload JSON data to S3 with enhanced error handling
        """
        try:
            # Ensure the key does not start with a /
            s3_key = s3_key.lstrip('/')
            
            # Convert data to JSON if it's not already a string
            if not isinstance(data, str):
                json_data = json.dumps(data, indent=2, default=str)
            else:
                json_data = data

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            self.logger.info(f"Successfully uploaded data to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            self.logger.error(f"Error uploading data to S3: {e}")
            import traceback
            traceback.print_exc()
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3
        
        Args:
            s3_key (str): S3 key of the file to download
            local_path (str): Local destination path
        
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            self.logger.info(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error downloading file from S3: {e}")
            return False

    def list_files(self, prefix: str = '') -> list:
        """
        List files in the bucket with an optional prefix
        
        Args:
            prefix (str, optional): Prefix to filter files
        
        Returns:
            list: List of file keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            self.logger.error(f"Error listing files in S3: {e}")
            return []

    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3
        
        Args:
            s3_key (str): S3 key to check
        
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            # If a 404 error is returned, the file does not exist
            if e.response['Error']['Code'] == '404':
                return False
            # If any other error occurs, log it and return False
            self.logger.error(f"Error checking file existence in S3: {e}")
            return False

    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            s3_key (str): S3 key of the file to delete
        
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            self.logger.info(f"Successfully deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting file from S3: {e}")
            return False

    def generate_presigned_url(self, s3_key: str, expiration=3600) -> str:
        """
        Generate a presigned URL for a file in S3
        
        Args:
            s3_key (str): S3 key of the file
            expiration (int, optional): URL expiration time in seconds. Defaults to 1 hour.
        
        Returns:
            str: Presigned URL or empty string if generation fails
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            self.logger.error(f"Error generating presigned URL: {e}")
            return ""

    def copy_file(self, source_key: str, destination_key: str) -> bool:
        """
        Copy a file within the same S3 bucket
        
        Args:
            source_key (str): Original S3 key
            destination_key (str): Destination S3 key
        
        Returns:
            bool: True if copy successful, False otherwise
        """
        try:
            copy_source = {
                'Bucket': self.bucket_name,
                'Key': source_key
            }
            self.s3_client.copy(copy_source, self.bucket_name, destination_key)
            self.logger.info(f"Successfully copied s3://{self.bucket_name}/{source_key} to s3://{self.bucket_name}/{destination_key}")
            return True
        except Exception as e:
            self.logger.error(f"Error copying file in S3: {e}")
            return False

    def get_file_metadata(self, s3_key: str) -> dict:
        """
        Retrieve metadata for a file in S3
        
        Args:
            s3_key (str): S3 key of the file
        
        Returns:
            dict: File metadata or empty dict if retrieval fails
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                'content_length': response.get('ContentLength', 0),
                'content_type': response.get('ContentType', ''),
                'last_modified': response.get('LastModified', None),
                'etag': response.get('ETag', ''),
                'version_id': response.get('VersionId', '')
            }
        except Exception as e:
            self.logger.error(f"Error retrieving file metadata: {e}")
            return {}

    def create_folder(self, folder_key: str) -> bool:
        """
        Create a 'folder' in S3 (by creating an empty object with a trailing slash)
        
        Args:
            folder_key (str): Folder path in S3
        
        Returns:
            bool: True if folder creation successful, False otherwise
        """
        try:
            # Ensure the folder key ends with a slash
            if not folder_key.endswith('/'):
                folder_key += '/'
            
            self.s3_client.put_object(Bucket=self.bucket_name, Key=folder_key, Body=b'')
            self.logger.info(f"Successfully created folder: s3://{self.bucket_name}/{folder_key}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating folder in S3: {e}")
            return False

    def __str__(self):
        """
        String representation of the S3 Handler
        
        Returns:
            str: Description of the S3 bucket
        """
        return f"S3 Bucket Handler: {self.bucket_name} (Region: {self.region_name})"

    def __repr__(self):
        """
        Detailed representation of the S3 Handler
        
        Returns:
            str: Detailed description of the S3 bucket
        """
        return f"S3Handler(bucket_name='{self.bucket_name}', region_name='{self.region_name}')"
        