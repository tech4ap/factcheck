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
    setup_logging, add_aws_arguments, add_common_arguments, 
    create_predictor_from_args, validate_and_load_models,
    print_prediction_result, save_prediction_result, 
    validate_s3_url_and_credentials, get_s3_file_info,
    cleanup_temporary_file, handle_prediction_error,
    create_enhanced_examples_text, PredictionContext
)
from utils.s3_utils import is_s3_url, setup_aws_credentials_from_env

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def detect_deepfake_s3(s3_url: str, 
                      aws_access_key_id: str = None,
                      aws_secret_access_key: str = None,
                      aws_session_token: str = None,
                      aws_region: str = 'us-east-1',
                      confidence_threshold: float = 0.5,
                      models_dir: str = 'models',
                      output_file: str = None,
                      cleanup: bool = True):
    """
    Detect deepfake in S3 file using modular functions.
    
    Args:
        s3_url: S3 URL of the file to analyze
        aws_access_key_id: AWS Access Key ID
        aws_secret_access_key: AWS Secret Access Key
        aws_session_token: AWS Session Token (for temporary credentials)
        aws_region: AWS region
        confidence_threshold: Confidence threshold for classification
        models_dir: Directory containing trained models
        output_file: Optional output file for results
        cleanup: Whether to cleanup downloaded files
        
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"🔍 Starting deepfake detection for S3 file: {s3_url}")
    
    # Validate S3 URL and credentials
    validate_s3_url_and_credentials(s3_url, aws_access_key_id)
    
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
            s3_url, aws_access_key_id, aws_secret_access_key, 
            aws_session_token, aws_region
        )
        
        # Run prediction with automatic cleanup
        with PredictionContext(predictor, cleanup_files=cleanup) as ctx:
            result = ctx.predict(s3_url, confidence_threshold)
            
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
    """Main function with modular argument parsing."""
    parser = argparse.ArgumentParser(
        description='Deepfake Detection for S3 Files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=create_enhanced_examples_text('predict_s3_deepfake.py'))
    
    # Add S3 URL as positional argument
    parser.add_argument('s3_url', type=str, nargs='?',
                       help='S3 URL of the file to analyze (e.g., s3://bucket/path/file.jpg)')
    
    # Add common arguments
    add_common_arguments(parser)
    
    # Add AWS arguments
    add_aws_arguments(parser, 'AWS S3 Configuration')
    
    # Add S3-specific arguments
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Do not cleanup downloaded temporary files')
    parser.add_argument('--setup-aws', action='store_true',
                       help='Show AWS credentials setup instructions')
    
    args = parser.parse_args()
    
    # Show AWS setup instructions if requested
    if args.setup_aws:
        setup_aws_credentials_from_env()
        return
    
    # Check if S3 URL is provided
    if not args.s3_url:
        logger.error("❌ S3 URL is required!")
        logger.info("💡 Use --setup-aws for AWS credentials setup instructions")
        logger.info("💡 Example: python predict_s3_deepfake.py s3://bucket/file.jpg")
        return
    
    try:
        # Run detection with enhanced error handling
        result = detect_deepfake_s3(
            s3_url=args.s3_url,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_session_token=args.aws_session_token,
            aws_region=args.aws_region,
            confidence_threshold=args.threshold,
            models_dir=args.models_dir,
            output_file=args.output,
            cleanup=not args.no_cleanup
        )
        
        # Summary
        logger.info("🎉 Deepfake detection completed successfully!")
        logger.info(f"📊 Result: {result['label']} (confidence: {result['confidence']:.1%})")
        
    except Exception as e:
        # Error handling is now done in detect_deepfake_s3 via common functions
        sys.exit(1)

if __name__ == "__main__":
    main() 