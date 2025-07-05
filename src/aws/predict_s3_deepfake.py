#!/usr/bin/env python3
"""
S3 Deepfake Detection Script

This script provides a simplified interface for detecting deepfakes
in files stored on Amazon S3.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from inference.predict_deepfake import EnhancedDeepfakePredictor
from inference.common import (
    setup_logging, add_common_arguments, 
    validate_and_load_models,
    print_prediction_result, save_prediction_result, 
    validate_s3_url_and_credentials, get_s3_file_info,
    handle_prediction_error, PredictionContext
)
from utils.s3_utils import is_s3_url
from core.config import get_config

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def detect_deepfake_s3(sqs_to_ml_url: str, 
                      aws_access_key_id: str = None,
                      aws_secret_access_key: str = None,
                      aws_session_token: str = None,
                      aws_region: str = None,
                      confidence_threshold: float = 0.5,
                      models_dir: str = 'models',
                      output_file: str = None,
                      cleanup: bool = True,
                      config_file: str = None):
    """
    Detect deepfake in S3 file using AWS credentials from config.
    
    Args:
        sqs_to_ml_url: S3 URL of the file to analyze
        aws_access_key_id: AWS Access Key ID (if None, loads from config/env)
        aws_secret_access_key: AWS Secret Access Key (if None, loads from config/env)
        aws_session_token: AWS Session Token (for temporary credentials)
        aws_region: AWS region (if None, loads from config/env, default: us-east-1)
        confidence_threshold: Confidence threshold for classification
        models_dir: Directory containing trained models
        output_file: Optional output file for results
        cleanup: Whether to cleanup downloaded files
        config_file: Path to config file (default: aws_config.json)
        
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"🔍 Starting deepfake detection for S3 file: {sqs_to_ml_url}")
    
    # Load configuration
    config = get_config(config_file or 'aws_config.json')
    
    # Use credentials from config if not provided
    aws_access_key_id = aws_access_key_id or config.aws.access_key_id
    aws_secret_access_key = aws_secret_access_key or config.aws.secret_access_key
    aws_session_token = aws_session_token or config.aws.session_token
    aws_region = aws_region or config.aws.region
    
    # Validate S3 URL and credentials
    validate_s3_url_and_credentials(sqs_to_ml_url, aws_access_key_id)
    
    # Create predictor
    predictor = EnhancedDeepfakePredictor(
        models_dir=models_dir,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        aws_region=aws_region
    )
    
    # Load and validate models
    validate_and_load_models(predictor)
    
    try:
        # Get S3 file info
        file_info = get_s3_file_info(
            sqs_to_ml_url, aws_access_key_id, aws_secret_access_key, 
            aws_session_token, aws_region
        )
        
        # Run prediction with automatic cleanup
        with PredictionContext(predictor, cleanup_files=cleanup) as ctx:
            result = ctx.predict(sqs_to_ml_url, confidence_threshold)
            
            # Add S3 file info to result
            if file_info:
                result['s3_file_info'] = file_info
            
            # Print results
            print_prediction_result(result)
            
            # Save results if requested
            if output_file:
                save_prediction_result(result, output_file, 'json')
            
            return result
        
    except Exception as e:
        handle_prediction_error(e, "S3 deepfake detection")
        raise

def main():
    """Main function with config-based AWS credentials."""
    parser = argparse.ArgumentParser(
        description='Deepfake Detection for S3 Files - Uses Config/Environment AWS Credentials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_s3_deepfake.py s3://bucket/image.jpg
  python predict_s3_deepfake.py s3://bucket/video.mp4 --threshold 0.8
  python predict_s3_deepfake.py s3://bucket/audio.wav --output results.json
  python predict_s3_deepfake.py s3://bucket/file.wav --config my_aws_config.json
  
Note: AWS credentials are loaded from aws_config.json file or environment variables.
Setup: Copy src/aws/aws_config_template.json to aws_config.json and add your credentials.
""")
    
    # Add S3 URL as positional argument
    parser.add_argument('sqs_to_ml_url', type=str, nargs='?',
                       help='S3 URL of the file to analyze (e.g., s3://bucket/path/file.jpg)')
    
    # Add common arguments
    add_common_arguments(parser)
    
    # Add S3-specific arguments
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Do not cleanup downloaded temporary files')
    parser.add_argument('--config', type=str, default='aws_config.json',
                       help='Path to configuration file (default: aws_config.json)')
    
    args = parser.parse_args()
    
    # Fixed AWS credentials - no setup needed
    
    # Check if S3 URL is provided
    if not args.sqs_to_ml_url:
        logger.error("❌ S3 URL is required!")
        logger.info("💡 AWS credentials are loaded from aws_config.json or environment variables")
        logger.info("💡 Setup: Copy src/aws/aws_config_template.json to aws_config.json and add your credentials")
        logger.info("💡 Example: python predict_s3_deepfake.py s3://bucket/file.jpg")
        return
    
    try:
        # Run detection with config-based AWS credentials
        result = detect_deepfake_s3(
            sqs_to_ml_url=args.sqs_to_ml_url,
            confidence_threshold=args.threshold,
            models_dir=args.models_dir,
            output_file=args.output,
            cleanup=not args.no_cleanup,
            config_file=args.config
        )
        
        # Summary
        logger.info("🎉 Deepfake detection completed successfully!")
        logger.info(f"📊 Result: {result['label']} (confidence: {result['confidence']:.1%})")
        
    except Exception as e:
        # Error handling is now done in detect_deepfake_s3 via common functions
        sys.exit(1)

if __name__ == "__main__":
    main() 