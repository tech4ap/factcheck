"""
S3 Utilities for Deepfake Detection

This module provides functionality to download files from S3 URLs
for processing with the deepfake detection system.
"""

import os
import boto3
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
import requests
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

class S3FileHandler:
    """Handle S3 file operations for deepfake detection."""
    
    def __init__(self, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_session_token: Optional[str] = None,
                 region_name: str = 'us-east-1'):
        """
        Initialize S3 client.
        
        Args:
            aws_access_key_id: AWS Access Key ID (optional, can use env vars)
            aws_secret_access_key: AWS Secret Access Key (optional, can use env vars)
            aws_session_token: AWS Session Token for temporary credentials (optional)
            region_name: AWS region name
        """
        self.region_name = region_name
        self.temp_dir = Path(tempfile.gettempdir()) / "deepfake_s3_cache"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize S3 client
        try:
            session_kwargs = {'region_name': region_name}
            
            if aws_access_key_id and aws_secret_access_key:
                session_kwargs.update({
                    'aws_access_key_id': aws_access_key_id,
                    'aws_secret_access_key': aws_secret_access_key
                })
                if aws_session_token:
                    session_kwargs['aws_session_token'] = aws_session_token
            
            self.session = boto3.Session(**session_kwargs)
            self.s3_client = self.session.client('s3')
            
            # Test connection
            self.s3_client.list_buckets()
            logger.info("✅ S3 client initialized successfully")
            
        except NoCredentialsError:
            logger.warning("⚠️ AWS credentials not found. Trying environment variables...")
            try:
                self.s3_client = boto3.client('s3', region_name=region_name)
                self.s3_client.list_buckets()
                logger.info("✅ S3 client initialized with environment credentials")
            except Exception as e:
                logger.error(f"❌ Failed to initialize S3 client: {e}")
                self.s3_client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {e}")
            self.s3_client = None
    
    def parse_s3_url(self, s3_url: str) -> Tuple[str, str]:
        """
        Parse S3 URL to extract bucket and key.
        
        Args:
            s3_url: S3 URL (s3://bucket/key or https://bucket.s3.region.amazonaws.com/key)
            
        Returns:
            Tuple of (bucket_name, object_key)
        """
        parsed = urlparse(s3_url)
        
        if parsed.scheme == 's3':
            # s3://bucket/key format
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
        elif parsed.scheme in ['http', 'https']:
            # https://bucket.s3.region.amazonaws.com/key format
            if '.s3.' in parsed.netloc or '.s3-' in parsed.netloc:
                bucket = parsed.netloc.split('.')[0]
                key = parsed.path.lstrip('/')
            else:
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
        else:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        
        if not bucket or not key:
            raise ValueError(f"Could not parse bucket and key from URL: {s3_url}")
        
        return bucket, key
    
    def download_file(self, s3_url: str, local_path: Optional[str] = None) -> str:
        """
        Download file from S3.
        
        Args:
            s3_url: S3 URL of the file
            local_path: Local path to save file (optional, uses temp dir if not provided)
            
        Returns:
            Path to downloaded file
        """
        if self.s3_client is None:
            raise RuntimeError("S3 client not initialized. Check AWS credentials.")
        
        bucket, key = self.parse_s3_url(s3_url)
        
        # Determine local file path
        if local_path is None:
            filename = Path(key).name
            local_path = self.temp_dir / filename
        else:
            local_path = Path(local_path)
        
        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"📥 Downloading from S3: {bucket}/{key}")
            
            # Check if file exists in S3
            self.s3_client.head_object(Bucket=bucket, Key=key)
            
            # Download file
            self.s3_client.download_file(bucket, key, str(local_path))
            
            logger.info(f"✅ Downloaded to: {local_path}")
            return str(local_path)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise FileNotFoundError(f"File not found in S3: {s3_url}")
            else:
                raise RuntimeError(f"S3 error ({error_code}): {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to download from S3: {e}")
    
    def get_file_info(self, s3_url: str) -> dict:
        """
        Get file information from S3.
        
        Args:
            s3_url: S3 URL of the file
            
        Returns:
            Dictionary with file information
        """
        if self.s3_client is None:
            raise RuntimeError("S3 client not initialized. Check AWS credentials.")
        
        bucket, key = self.parse_s3_url(s3_url)
        
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            
            return {
                'bucket': bucket,
                'key': key,
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'etag': response.get('ETag', '').strip('"')
            }
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise FileNotFoundError(f"File not found in S3: {s3_url}")
            else:
                raise RuntimeError(f"S3 error ({error_code}): {e}")
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Clean up temporary files older than specified hours.
        
        Args:
            max_age_hours: Maximum age of files to keep (in hours)
        """
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        if not self.temp_dir.exists():
            return
        
        cleaned_count = 0
        for file_path in self.temp_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Could not delete {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} temporary files")

def download_from_s3(s3_url: str, 
                    aws_access_key_id: Optional[str] = None,
                    aws_secret_access_key: Optional[str] = None,
                    aws_session_token: Optional[str] = None,
                    region_name: str = 'us-east-1') -> str:
    """
    Convenience function to download a file from S3.
    
    Args:
        s3_url: S3 URL of the file
        aws_access_key_id: AWS Access Key ID (optional)
        aws_secret_access_key: AWS Secret Access Key (optional)
        aws_session_token: AWS Session Token (optional)
        region_name: AWS region name
        
    Returns:
        Path to downloaded file
    """
    handler = S3FileHandler(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name
    )
    
    return handler.download_file(s3_url)

def is_s3_url(url: str) -> bool:
    """
    Check if a URL is an S3 URL.
    
    Args:
        url: URL to check
        
    Returns:
        True if it's an S3 URL, False otherwise
    """
    if not url:
        return False
    
    parsed = urlparse(url)
    
    # s3://bucket/key format
    if parsed.scheme == 's3':
        return True
    
    # https://bucket.s3.region.amazonaws.com/key format
    if parsed.scheme in ['http', 'https'] and '.s3.' in parsed.netloc:
        return True
    
    return False

def get_file_extension_from_s3_url(s3_url: str) -> str:
    """
    Extract file extension from S3 URL.
    
    Args:
        s3_url: S3 URL
        
    Returns:
        File extension (e.g., '.jpg', '.mp4')
    """
    try:
        _, key = S3FileHandler().parse_s3_url(s3_url)
        return Path(key).suffix.lower()
    except:
        return ''

# AWS credential helpers
def setup_aws_credentials_from_env():
    """
    Setup AWS credentials from environment variables.
    Prints instructions if credentials are not found.
    """
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("⚠️  AWS credentials not found in environment variables.")
        print("\n🔧 To setup AWS credentials, choose one of these options:")
        print("\n1️⃣  Set environment variables:")
        print("   export AWS_ACCESS_KEY_ID='your_access_key'")
        print("   export AWS_SECRET_ACCESS_KEY='your_secret_key'")
        print("   export AWS_DEFAULT_REGION='us-east-1'  # optional")
        print("\n2️⃣  Use AWS CLI configuration:")
        print("   aws configure")
        print("\n3️⃣  Use AWS IAM roles (if running on EC2)")
        print("\n4️⃣  Pass credentials directly to the functions")
        
        return False
    else:
        print("✅ AWS credentials found in environment variables")
        return True 