import cv2
import numpy as np
import easyocr
import argparse
from pathlib import Path

def detect_and_remove_text(image_path, output_path, confidence_threshold=0.3, dilation_kernel_size=3):
    """
    Detect and remove text from images using OCR and inpainting
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return False
    
    height, width = img.shape[:2]
    
    # Create mask for text regions
    mask = np.zeros((height, width), dtype=np.uint8)
    text_detected = False
    
    try:
        # Detect text with multiple confidence thresholds for robustness
        results = reader.readtext(str(image_path))
        
        for (bbox, text, confidence) in results:
            if confidence > confidence_threshold:
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                
                # Add padding around text to ensure complete removal
                padding = 5
                x1 = max(0, top_left[0] - padding)
                y1 = max(0, top_left[1] - padding)
                x2 = min(width, bottom_right[0] + padding)
                y2 = min(height, bottom_right[1] + padding)
                
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                text_detected = True
                
                print(f"Detected text: '{text}' with confidence {confidence:.2f} at [{x1},{y1},{x2},{y2}]")
                
    except Exception as e:
        print(f"Text detection failed for {image_path}: {e}")
        return False
    
    if not text_detected:
        print(f"No text detected in {image_path.name}")
        # Copy original image if no text detected
        cv2.imwrite(str(output_path), img)
        return True
    
    # Apply dilation to the mask to ensure complete text removal
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Apply inpainting only to text regions
    result = cv2.inpaint(img, dilated_mask, 5, cv2.INPAINT_TELEA)
    
    # Save result
    cv2.imwrite(str(output_path), result)
    print(f"Processed: {image_path.name} -> {output_path.name}")
    
    # Optional: Save mask for debugging
    debug_dir = output_path.parent / "debug_masks"
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / f"mask_{image_path.name}"), dilated_mask)
    
    return True

def process_bird_dataset(root_dir, confidence_threshold=0.3):
    """Process all bird images in the directory structure"""
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Root directory does not exist: {root_dir}")
        return
    
    # Find all class directories
    class_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        images_dir = class_dir / 'images' / 'train'
        
        if images_dir.exists():
            print(f"\nProcessing {class_dir.name}...")
            
            # Find all image files
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(images_dir.glob(ext))
            
            processed_count = 0
            total_count = len(image_files)
            
            for image_path in image_files:
                output_path = images_dir / f"clean_{image_path.name}"
                
                # Skip if already processed
                if output_path.exists():
                    print(f"Skipping already processed: {image_path.name}")
                    continue
                
                success = detect_and_remove_text(image_path, output_path, confidence_threshold)
                if success:
                    processed_count += 1
                
                # Progress update
                if processed_count % 10 == 0:
                    print(f"  Progress: {processed_count}/{total_count}")
            
            print(f"  Completed: {processed_count}/{total_count} images processed")

def enhance_text_detection(image_path):
    """
    Pre-process image to enhance text detection for various colors
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple thresholds to detect text of different colors
    _, thresh1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)  # Dark text
    _, thresh2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # Light text
    
    # Combine thresholds
    combined = cv2.bitwise_or(thresh1, thresh2)
    
    # Apply morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

if __name__ == "__main__":
    # Initialize EasyOCR reader with more languages and better settings
    print("Loading EasyOCR model...")
    reader = easyocr.Reader(
        ['en'],  # English
        gpu=True,  # Use GPU if available
        model_storage_directory='./models',
        download_enabled=True
    )
    print("EasyOCR model loaded!")
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Remove text from bird images using advanced text detection')
    parser.add_argument('root_dir', type=str, help='Root directory containing bird class folders')
    parser.add_argument('--confidence', type=float, default=0.3, 
                       help='Confidence threshold for text detection (default: 0.3)')
    parser.add_argument('--dilation', type=int, default=3,
                       help='Dilation kernel size for mask expansion (default: 3)')
    
    args = parser.parse_args()
    
    print(f"Starting text removal with confidence threshold: {args.confidence}")
    process_bird_dataset(args.root_dir, args.confidence)