"""
Optimized Image Processing Pipeline

This is a refactored version of the image pipeline that demonstrates integration
with the new core modules, eliminating code duplication and improving structure.
"""

import cv2
import numpy as np
import time
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Import from new core modules
from src.core import (
    get_config, get_logger, BaseProcessor, 
    IMAGE_EXTENSIONS, DEFAULT_IMAGE_SIZE,
    robust_processing, log_execution, measure_performance
)
from src.core.exceptions import DataError, ValidationError

class OptimizedImageProcessor(BaseProcessor):
    """
    Optimized image processor using the new core architecture.
    
    Features:
    - Inherits common functionality from BaseProcessor
    - Uses centralized configuration and logging
    - Applies consistent error handling and performance tracking
    - Supports face detection and image enhancement
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the image processor.
        
        Args:
            config: Optional configuration override
        """
        super().__init__('image', IMAGE_EXTENSIONS)
        
        # Get global configuration
        self.config = get_config()
        if config:
            # Override with custom config
            self.config = self.config.override(**config)
        
        # Image-specific settings
        self.target_size = self.config.model.image_size
        self.face_cascade = None
        self._load_face_detector()
    
    def _load_face_detector(self) -> None:
        """Load face detection model."""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.logger.info("Face detector loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load face detector: {e}")
    
    @robust_processing(extensions=IMAGE_EXTENSIONS)
    @measure_performance
    def process_file(self, input_path: Path, output_path: Path, 
                    detect_faces: bool = False, enhance: bool = True, **kwargs) -> bool:
        """
        Process a single image file.
        
        Args:
            input_path: Input image path
            output_path: Output image path
            detect_faces: Whether to perform face detection
            enhance: Whether to enhance image quality
            **kwargs: Additional processing parameters
            
        Returns:
            True if processing succeeded
        """
        with self.track_processing(input_path):
            try:
                # Load and validate image
                image = self._load_image(input_path)
                
                # Apply enhancements if requested
                if enhance:
                    image = self._enhance_image(image)
                
                # Detect faces if requested
                faces = []
                if detect_faces and self.face_cascade is not None:
                    faces = self._detect_faces(image)
                
                # Resize to target size
                processed_image = self._resize_image(image, self.target_size)
                
                # Save processed image
                self._save_image(processed_image, output_path)
                
                # Save face information if detected
                if faces:
                    self._save_face_info(faces, output_path)
                
                self.logger.debug(f"Processed image: {input_path} -> {output_path}")
                return True
                
            except Exception as e:
                raise DataError(f"Failed to process image: {e}", 
                              file_path=str(input_path), operation="process")
    
    @log_execution(level='DEBUG')
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image from file with validation.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array
            
        Raises:
            DataError: If image cannot be loaded
        """
        try:
            # Try PIL first for broader format support
            pil_image = Image.open(image_path).convert('RGB')
            image = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV compatibility
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if image.size == 0:
                raise ValueError("Empty image")
            
            return image
            
        except Exception as e:
            raise DataError(f"Cannot load image: {e}", 
                          file_path=str(image_path), operation="load")
    
    @log_execution(level='DEBUG')
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply image enhancement techniques.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Apply histogram equalization to improve contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply bilateral filter for noise reduction while preserving edges
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Enhancement failed: {e}")
            return image  # Return original if enhancement fails
    
    @log_execution(level='DEBUG')
    def _detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of face detection results
        """
        if self.face_cascade is None:
            return []
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Convert to list of dictionaries
            face_data = []
            for i, (x, y, w, h) in enumerate(faces):
                face_data.append({
                    'id': i,
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': 1.0  # Haar cascades don't provide confidence
                })
            
            self.logger.debug(f"Detected {len(face_data)} faces")
            return face_data
            
        except Exception as e:
            self.logger.warning(f"Face detection failed: {e}")
            return []
    
    @log_execution(level='DEBUG')
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            
        Returns:
            Resized image
        """
        try:
            # Use high-quality interpolation for resizing
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            return resized
            
        except Exception as e:
            raise DataError(f"Failed to resize image: {e}", operation="resize")
    
    @log_execution(level='DEBUG')
    def _save_image(self, image: np.ndarray, output_path: Path) -> None:
        """
        Save image to file.
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with high quality
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        except Exception as e:
            raise DataError(f"Failed to save image: {e}", 
                          file_path=str(output_path), operation="save")
    
    def _save_face_info(self, faces: List[Dict], image_path: Path) -> None:
        """
        Save face detection information.
        
        Args:
            faces: List of face detection results
            image_path: Original image path (used to generate metadata filename)
        """
        if not faces:
            return
        
        try:
            import json
            
            # Create metadata filename
            metadata_path = image_path.parent / f"{image_path.stem}_faces.json"
            
            # Save face data
            with open(metadata_path, 'w') as f:
                json.dump({
                    'image_file': image_path.name,
                    'face_count': len(faces),
                    'faces': faces,
                    'detection_timestamp': time.time()
                }, f, indent=2)
            
            self.logger.debug(f"Saved face metadata: {metadata_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save face metadata: {e}")
    
    @log_execution(level='INFO')
    def batch_process_with_faces(self, input_dir: Path, output_dir: Path,
                                detect_faces: bool = True, 
                                enhance: bool = True) -> Dict[str, int]:
        """
        Process a batch of images with face detection.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for processed images
            detect_faces: Whether to detect faces
            enhance: Whether to enhance images
            
        Returns:
            Dictionary with processing statistics
        """
        # Find all image files
        image_files = self.find_files(input_dir)
        
        if not image_files:
            self.logger.warning(f"No image files found in {input_dir}")
            return {'processed': 0, 'failed': 0, 'total': 0}
        
        self.logger.info(f"Processing {len(image_files)} images...")
        
        # Process files
        successful, failed = self.process_batch(
            image_files, output_dir, 
            detect_faces=detect_faces, enhance=enhance
        )
        
        # Get processing statistics
        stats = self.get_stats()
        
        results = {
            'processed': successful,
            'failed': failed,
            'total': len(image_files),
            'processing_time': stats['processing_time'],
            'avg_time_per_image': stats['processing_time'] / len(image_files) if image_files else 0
        }
        
        self.logger.info(f"Batch processing complete: {results}")
        return results
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats."""
        return self.supported_extensions.copy()
    
    def validate_image_quality(self, image_path: Path) -> Dict[str, Union[bool, float]]:
        """
        Validate image quality metrics.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            image = self._load_image(image_path)
            
            # Calculate basic quality metrics
            height, width = image.shape[:2]
            file_size = image_path.stat().st_size
            
            # Calculate image sharpness (Laplacian variance)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            
            quality_metrics = {
                'is_valid': True,
                'width': width,
                'height': height,
                'file_size_mb': file_size / (1024 * 1024),
                'sharpness': float(sharpness),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'aspect_ratio': width / height,
                'meets_min_size': width >= 64 and height >= 64,
                'meets_target_size': (width, height) == self.target_size
            }
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Quality validation failed for {image_path}: {e}")
            return {
                'is_valid': False,
                'error': str(e)
            }

# Convenience functions for backward compatibility and ease of use
@log_execution()
def process_image_directory(input_dir: str, output_dir: str, **kwargs) -> Dict[str, int]:
    """
    Process all images in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        **kwargs: Additional processing options
        
    Returns:
        Processing statistics
    """
    processor = OptimizedImageProcessor()
    return processor.batch_process_with_faces(
        Path(input_dir), Path(output_dir), **kwargs
    )

@log_execution()  
def validate_image_dataset(data_dir: str) -> Dict[str, any]:
    """
    Validate an image dataset.
    
    Args:
        data_dir: Dataset directory path
        
    Returns:
        Validation report
    """
    processor = OptimizedImageProcessor()
    data_path = Path(data_dir)
    
    # Find all images
    image_files = processor.find_files(data_path)
    
    validation_results = {
        'total_files': len(image_files),
        'valid_files': 0,
        'invalid_files': 0,
        'quality_metrics': {
            'avg_sharpness': 0,
            'avg_brightness': 0,
            'avg_contrast': 0
        },
        'file_issues': []
    }
    
    if not image_files:
        return validation_results
    
    # Validate each image
    quality_sum = {'sharpness': 0, 'brightness': 0, 'contrast': 0}
    
    for image_path in image_files:
        metrics = processor.validate_image_quality(image_path)
        
        if metrics.get('is_valid', False):
            validation_results['valid_files'] += 1
            quality_sum['sharpness'] += metrics.get('sharpness', 0)
            quality_sum['brightness'] += metrics.get('brightness', 0)
            quality_sum['contrast'] += metrics.get('contrast', 0)
        else:
            validation_results['invalid_files'] += 1
            validation_results['file_issues'].append({
                'file': str(image_path),
                'error': metrics.get('error', 'Unknown error')
            })
    
    # Calculate averages
    if validation_results['valid_files'] > 0:
        validation_results['quality_metrics'] = {
            'avg_sharpness': quality_sum['sharpness'] / validation_results['valid_files'],
            'avg_brightness': quality_sum['brightness'] / validation_results['valid_files'],
            'avg_contrast': quality_sum['contrast'] / validation_results['valid_files']
        }
    
    return validation_results

if __name__ == "__main__":
    import argparse
    
    # Set up command line interface
    parser = argparse.ArgumentParser(description="Optimized Image Processing Pipeline")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", help="Output directory for processed images")
    parser.add_argument("--no-faces", action="store_true", help="Skip face detection")
    parser.add_argument("--no-enhance", action="store_true", help="Skip image enhancement")
    parser.add_argument("--validate-only", action="store_true", help="Only validate dataset")
    
    args = parser.parse_args()
    
    # Configure logging for command line usage
    from src.core import setup_logging
    setup_logging(level='INFO', colored_console=True)
    
    logger = get_logger(__name__)
    
    if args.validate_only:
        # Validation mode
        logger.info(f"Validating image dataset: {args.input_dir}")
        results = validate_image_dataset(args.input_dir)
        
        print(f"\n📊 Dataset Validation Results:")
        print(f"   Total files: {results['total_files']}")
        print(f"   Valid files: {results['valid_files']}")
        print(f"   Invalid files: {results['invalid_files']}")
        
        if results['valid_files'] > 0:
            metrics = results['quality_metrics']
            print(f"\n📈 Quality Metrics (averages):")
            print(f"   Sharpness: {metrics['avg_sharpness']:.2f}")
            print(f"   Brightness: {metrics['avg_brightness']:.2f}")
            print(f"   Contrast: {metrics['avg_contrast']:.2f}")
    
    else:
        # Processing mode
        logger.info(f"Processing images: {args.input_dir} -> {args.output_dir}")
        results = process_image_directory(
            args.input_dir, 
            args.output_dir,
            detect_faces=not args.no_faces,
            enhance=not args.no_enhance
        )
        
        print(f"\n✅ Processing Complete:")
        print(f"   Processed: {results['processed']}")
        print(f"   Failed: {results['failed']}")
        print(f"   Total time: {results['processing_time']:.2f}s")
        print(f"   Avg per image: {results['avg_time_per_image']:.2f}s") 