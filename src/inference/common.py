"""
Common utilities for deepfake detection inference scripts.

This module provides shared functionality used by both the general inference
script and the S3-specific script to eliminate code duplication.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Any, Union
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def add_aws_arguments(parser: argparse.ArgumentParser, group_title: str = 'AWS S3 Options') -> argparse.ArgumentParser:
    """
    Add AWS credential arguments to argument parser.
    
    Args:
        parser: ArgumentParser instance
        group_title: Title for the argument group
        
    Returns:
        ArgumentParser with AWS arguments added
    """
    aws_group = parser.add_argument_group(group_title)
    aws_group.add_argument('--aws-access-key-id', type=str,
                          help='AWS Access Key ID (or use AWS_ACCESS_KEY_ID env var)')
    aws_group.add_argument('--aws-secret-access-key', type=str,
                          help='AWS Secret Access Key (or use AWS_SECRET_ACCESS_KEY env var)')
    aws_group.add_argument('--aws-session-token', type=str,
                          help='AWS Session Token for temporary credentials')
    aws_group.add_argument('--aws-region', type=str, default='us-east-1',
                          help='AWS region (default: us-east-1)')
    return parser

def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common prediction arguments to argument parser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        ArgumentParser with common arguments added
    """
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained models (default: models)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for classification (default: 0.5)')
    parser.add_argument('--output', type=str,
                       help='Output file for results')
    return parser

def create_predictor_from_args(args: argparse.Namespace) -> 'EnhancedDeepfakePredictor':
    """
    Create EnhancedDeepfakePredictor from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configured EnhancedDeepfakePredictor instance
    """
    # Import here to avoid circular imports
    try:
        from .predict_deepfake import EnhancedDeepfakePredictor
    except ImportError:
        # Handle direct execution
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from predict_deepfake import EnhancedDeepfakePredictor
    
    return EnhancedDeepfakePredictor(
        models_dir=args.models_dir,
        aws_access_key_id=getattr(args, 'aws_access_key_id', None),
        aws_secret_access_key=getattr(args, 'aws_secret_access_key', None),
        aws_session_token=getattr(args, 'aws_session_token', None),
        aws_region=getattr(args, 'aws_region', 'us-east-1')
    )

def validate_and_load_models(predictor: 'EnhancedDeepfakePredictor', 
                           use_best: bool = True) -> Dict[str, bool]:
    """
    Validate and load models with comprehensive error handling.
    
    Args:
        predictor: EnhancedDeepfakePredictor instance
        use_best: Whether to use best models instead of final models
        
    Returns:
        Dictionary of loaded models
        
    Raises:
        RuntimeError: If no models could be loaded
    """
    loaded_models = predictor.load_models(use_best=use_best)
    
    if not any(loaded_models.values()):
        raise RuntimeError(
            "No models could be loaded! Make sure you have trained models in the models directory. "
            "You can train models using: python train_and_save_models.py --demo"
        )
    
    loaded_model_names = [k for k, v in loaded_models.items() if v]
    logger.info(f"Loaded models: {loaded_model_names}")
    
    return loaded_models

def print_prediction_result(result: Dict[str, Any]) -> None:
    """
    Print formatted prediction result with enhanced display.
    
    Args:
        result: Prediction result dictionary
    """
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION RESULTS")
    print("="*60)
    print(f"File: {result['file_path']}")
    
    # Show S3 information if applicable
    if result.get('is_s3_file', False):
        print(f"Source: Amazon S3")
        if result.get('local_path'):
            print(f"Downloaded to: {result['local_path']}")
        if result.get('s3_file_info'):
            s3_info = result['s3_file_info']
            print(f"S3 Info: {s3_info.get('size', 0):,} bytes, {s3_info.get('content_type', 'Unknown')}")
    
    print(f"Type: {result['file_type'].upper()}")
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Raw Score: {result['prediction_score']:.4f}")
    
    if result['label'] == 'FAKE':
        print("WARNING: This appears to be a DEEPFAKE!")
    else:
        print("This appears to be REAL content.")
    
    # Additional info based on file type
    if result['file_type'] == 'video':
        print(f"Duration: {result.get('duration_seconds', 'N/A'):.1f}s")
        print(f"Frames analyzed: {result.get('frames_extracted', 'N/A')}")
    elif result['file_type'] == 'audio':
        print(f"Duration: {result.get('duration_seconds', 'N/A'):.1f}s")
    
    print("="*60)

def save_prediction_result(result: Dict[str, Any], 
                         output_file: str, 
                         format_type: str = 'json') -> None:
    """
    Save prediction result to file in specified format.
    
    Args:
        result: Prediction result dictionary
        output_file: Output file path
        format_type: Output format ('json' or 'csv')
    """
    try:
        if format_type.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        elif format_type.lower() == 'csv':
            pd.DataFrame([result]).to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

def print_batch_summary(results_df: pd.DataFrame) -> None:
    """
    Print summary statistics for batch processing results.
    
    Args:
        results_df: DataFrame with batch processing results
    """
    total_files = len(results_df)
    
    if 'error' in results_df.columns:
        successful = len(results_df[results_df['error'].isna()])
        failed = total_files - successful
    else:
        successful = total_files
        failed = 0
    
    fake_count = len(results_df[results_df['label'] == 'FAKE']) if 'label' in results_df.columns else 0
    real_count = len(results_df[results_df['label'] == 'REAL']) if 'label' in results_df.columns else 0
    
    print(f"\nBATCH PROCESSING SUMMARY")
    print("="*40)
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {successful}")
    if failed > 0:
        print(f"Failed to process: {failed}")
    print(f"Detected as FAKE: {fake_count}")
    print(f"Detected as REAL: {real_count}")
    
    if 'confidence' in results_df.columns and len(results_df) > 0:
        avg_confidence = results_df['confidence'].mean()
        print(f"Average confidence: {avg_confidence:.1%}")

def list_available_models(predictor: 'EnhancedDeepfakePredictor') -> None:
    """
    List all available models with metadata.
    
    Args:
        predictor: EnhancedDeepfakePredictor instance
    """
    models = predictor.list_available_models()
    print("\nAVAILABLE MODELS:")
    print("="*50)
    
    if not models:
        print("No models found!")
        print("Train models using: python train_and_save_models.py --demo")
        return
    
    for model_type, metadata in models.items():
        print(f"Model: {model_type.upper()}")
        print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")
        
        # Get evaluation results
        eval_results = metadata.get('evaluation_results', {})
        if eval_results:
            accuracy = eval_results.get('accuracy', 'N/A')
            if isinstance(accuracy, (int, float)):
                accuracy = f"{accuracy:.2%}"
            print(f"   Accuracy: {accuracy}")
        else:
            print(f"   Accuracy: N/A")
        
        print(f"   Path: {metadata.get('model_path', 'N/A')}")
        print()

def validate_s3_url_and_credentials(sqs_to_ml_url: str, 
def validate_s3_url_and_credentials(sqs_to_ml_url: str, 
                                   aws_access_key_id: Optional[str] = None) -> None:
    """
    Validate S3 URL format and check for AWS credentials.
    
    Args:
        sqs_to_ml_url: S3 URL to validate
        sqs_to_ml_url: S3 URL to validate
        aws_access_key_id: AWS access key (if provided)
        
    Raises:
        ValueError: If S3 URL format is invalid
    """
    try:
        from ..utils.s3_utils import is_s3_url
    except ImportError:
        # Handle direct execution
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent
        sys.path.append(str(src_path))
        from utils.s3_utils import is_s3_url
    
    if not is_s3_url(sqs_to_ml_url):
    if not is_s3_url(sqs_to_ml_url):
        raise ValueError(
            f"Invalid S3 URL: {sqs_to_ml_url}\n"
            "S3 URLs should be in format: s3://bucket/path/to/file.ext"
        )
    
    # Check AWS credentials if not provided
    if not aws_access_key_id and not os.getenv('AWS_ACCESS_KEY_ID'):
        logger.warning("No AWS credentials provided!")
        logger.info("Use environment variables or pass credentials via command line")
        logger.info("Run with --setup-aws for setup instructions")

def cleanup_temporary_file(file_path: Optional[str]) -> None:
    """
    Clean up temporary file if it exists.
    
    Args:
        file_path: Path to temporary file
    """
    if file_path and Path(file_path).exists():
        try:
            Path(file_path).unlink()
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not cleanup temporary file: {e}")

def get_s3_file_info(sqs_to_ml_url: str, 
def get_s3_file_info(sqs_to_ml_url: str, 
                    aws_access_key_id: Optional[str] = None,
                    aws_secret_access_key: Optional[str] = None,
                    aws_session_token: Optional[str] = None,
                    aws_region: str = 'us-east-1') -> Optional[Dict[str, Any]]:
    """
    Get S3 file information safely with error handling.
    
    Args:
        sqs_to_ml_url: S3 URL of the file
        sqs_to_ml_url: S3 URL of the file
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        aws_session_token: AWS session token
        aws_region: AWS region
        
    Returns:
        S3 file information dictionary or None if failed
    """
    try:
        try:
            from ..utils.s3_utils import S3FileHandler
        except ImportError:
            # Handle direct execution
            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent
            sys.path.append(str(src_path))
            from utils.s3_utils import S3FileHandler
        
        s3_handler = S3FileHandler(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )
        
        if s3_handler.s3_client is None:
            logger.warning("S3 client not available - skipping S3 metadata")
            return None
        
        file_info = s3_handler.get_file_info(sqs_to_ml_url)
        logger.info(f"S3 File info: {file_info['size']:,} bytes, {file_info['content_type']}")
        return file_info
        
    except Exception as e:
        logger.warning(f"Could not get S3 file info: {e}")
        return None

def handle_prediction_error(error: Exception, context: str = "prediction") -> None:
    """
    Handle prediction errors with informative messages.
    
    Args:
        error: Exception that occurred
        context: Context where error occurred
    """
    error_msg = str(error)
    
    if "S3 client not initialized" in error_msg:
        logger.error("AWS credentials not configured properly")
        logger.info("Check your AWS credentials and try again")
    elif "No models could be loaded" in error_msg:
        logger.error("No trained models found")
        logger.info("Train models using: python train_and_save_models.py --demo")
    elif "File not found" in error_msg:
        logger.error(f"File not found: {error_msg}")
        logger.info("Check the file path and try again")
    else:
        logger.error(f"Error during {context}: {error}")
    
    # Log full traceback in debug mode
    logger.debug(f"Full error details:", exc_info=True)

def create_enhanced_examples_text(script_name: str, include_s3: bool = True) -> str:
    """
    Create enhanced examples text for help messages.
    
    Args:
        script_name: Name of the script for examples
        include_s3: Whether to include S3 examples
        
    Returns:
        Formatted examples text
    """
    examples = []
    
    if include_s3:
        examples.extend([
            f"# Detect file from S3",
            f"python {script_name} s3://bucket/file.jpg",
            f"",
            f"# S3 with custom credentials", 
            f"python {script_name} s3://bucket/file.mp4 \\",
            f"  --aws-access-key-id YOUR_KEY \\",
            f"  --aws-secret-access-key YOUR_SECRET",
            f"",
        ])
    
    examples.extend([
        f"# Detect local file",
        f"python {script_name} photo.jpg",
        f"", 
        f"# Custom confidence threshold",
        f"python {script_name} video.mp4 --threshold 0.8",
        f"",
        f"# Save results to file",
        f"python {script_name} audio.wav --output results.json",
    ])
    
    return "\n".join(examples)

class PredictionContext:
    """Context manager for prediction operations with cleanup."""
    
    def __init__(self, 
                 predictor: 'EnhancedDeepfakePredictor',
                 cleanup_files: bool = True):
        self.predictor = predictor
        self.cleanup_files = cleanup_files
        self.temp_files = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_files:
            for temp_file in self.temp_files:
                cleanup_temporary_file(temp_file)
    
    def add_temp_file(self, file_path: str):
        """Add a temporary file to be cleaned up."""
        self.temp_files.append(file_path)
    
    def predict(self, input_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Run prediction with automatic temp file tracking."""
        result = self.predictor.predict_auto(input_path, confidence_threshold)
        
        # Track temporary files for cleanup
        if result.get('local_path') and result.get('is_s3_file'):
            self.add_temp_file(result['local_path'])
        
        return result 