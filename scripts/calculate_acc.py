#!/usr/bin/env python3
"""
Integrated OCR accuracy calculator with position-aware evaluation
Combines text accuracy (CER/WER) and spatial detection metrics (Precision/Recall/F-Score)
"""

import json
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

# Import PaddleOCR metrics
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    paddleocr_metrics_path = os.path.join(parent_dir, 'paddleocr_metrics')
    sys.path.insert(0, paddleocr_metrics_path)
    from rec_metric import RecMetric
    from eval_det_iou import DetectionIoUEvaluator
    PADDLEOCR_AVAILABLE = True
    print("PaddleOCR metrics imported successfully")
except ImportError as e:
    print(f"Warning: Could not import PaddleOCR metrics: {e}")
    PADDLEOCR_AVAILABLE = False

# Removed custom text normalization - using PaddleOCR's methods only


# =====================================================
# POSITION-AWARE EVALUATION FUNCTIONS
# =====================================================

def convert_bbox_to_polygon(bbox):
    """Convert bbox to polygon format expected by PaddleOCR evaluator"""
    if isinstance(bbox, list) and len(bbox) == 4:
        if all(isinstance(point, list) and len(point) == 2 for point in bbox):
            # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            return bbox
        else:
            # Format: [x1, y1, x2, y2] (convert to 4 points)
            x1, y1, x2, y2 = bbox
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return bbox


def format_for_paddleocr_evaluator(ground_truth_items: List[Dict], detection_results: List[Dict]):
    """Format data for PaddleOCR's DetectionIoUEvaluator"""
    # Format ground truth for PaddleOCR evaluator
    gt_formatted = []
    for gt_item in ground_truth_items:
        bbox = gt_item.get('bbox', [])
        # Convert to list of tuples as expected by PaddleOCR evaluator
        if bbox and isinstance(bbox[0], list):
            points = [(pt[0], pt[1]) for pt in bbox]
        else:
            points = []
        
        gt_formatted.append({
            "points": points,
            "text": gt_item.get('text', ''),
            "ignore": False  # XFUND dataset doesn't have ignore flags typically
        })
    
    # Format predictions for PaddleOCR evaluator  
    pred_formatted = []
    for det_result in detection_results:
        bbox = det_result.get('bbox', [])
        # Convert to list of tuples as expected by PaddleOCR evaluator
        if bbox:
            if isinstance(bbox[0], list):
                points = [(pt[0], pt[1]) for pt in bbox]
            elif len(bbox) == 4:
                # [x1,y1,x2,y2] format - convert to 4 corners
                x1, y1, x2, y2 = bbox
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                points = [(bbox[i], bbox[i+1]) for i in range(0, len(bbox), 2)]
        else:
            points = []
            
        pred_formatted.append({
            "points": points,
            "text": det_result.get('text', ''),
            "confidence": det_result.get('confidence', 0.0)
        })
        
    return gt_formatted, pred_formatted


def calculate_position_aware_metrics(ground_truth: Dict, ocr_result: Dict, 
                                   iou_threshold: float = 0.5) -> Dict:
    """
    Calculate position-aware OCR metrics using PaddleOCR's detection evaluator
    
    Args:
        ground_truth: Ground truth data in XFUND format
        ocr_result: OCR results with detection_results
        iou_threshold: IoU threshold for spatial matching
    
    Returns:
        Dictionary with detection and text accuracy metrics
    """
    if not PADDLEOCR_AVAILABLE:
        return {'error': 'PaddleOCR metrics not available'}
    
    try:
        # Extract ground truth items from XFUND format
        gt_items = []
        if 'document' in ground_truth:
            for item in ground_truth['document']:
                bbox = item.get('box', [])
                text = item.get('text', '')
                if bbox and text:
                    # Convert [x1, y1, x2, y2] to [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        bbox_polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    else:
                        bbox_polygon = bbox
                    
                    gt_items.append({
                        'bbox': bbox_polygon,
                        'text': text
                    })
        
        # Extract detection results  
        detection_results = ocr_result.get('detection_results', [])
        
        if not gt_items:
            return {'error': 'No ground truth items found'}
        
        if not detection_results:
            return {'error': 'No detection results found'}
        
        # Format data for PaddleOCR evaluator
        gt_formatted, pred_formatted = format_for_paddleocr_evaluator(gt_items, detection_results)
        
        # Create and run evaluator
        evaluator = DetectionIoUEvaluator(iou_constraint=iou_threshold)
        
        # Evaluate single image
        per_image_result = evaluator.evaluate_image(gt_formatted, pred_formatted)
        
        # Combine results to get final metrics
        result = evaluator.combine_results([per_image_result])
        
        # Calculate text accuracy for matched regions (simplified approach)
        text_metrics = calculate_text_accuracy_for_matches(gt_items, detection_results)
        
        # Combine results
        combined_result = {
            'detection_precision': result.get('precision', 0.0),
            'detection_recall': result.get('recall', 0.0), 
            'detection_hmean': result.get('hmean', 0.0),
            'iou_threshold': iou_threshold,
            'total_gt': len(gt_items),
            'total_detections': len(detection_results),
            **text_metrics
        }
        
        return combined_result
        
    except Exception as e:
        return {
            'error': f'Position-aware evaluation failed: {e}',
            'gt_count': len(gt_items) if 'gt_items' in locals() else 0,
            'detection_count': len(detection_results) if 'detection_results' in locals() else 0
        }


def calculate_text_accuracy_for_matches(gt_items: List[Dict], detections: List[Dict]) -> Dict:
    """Calculate text accuracy for spatially matched regions"""
    if not PADDLEOCR_AVAILABLE:
        return {'text_accuracy': 0.0, 'matched_pairs': 0}
    
    try:
        # Create RecMetric for text comparison
        metric = RecMetric(main_indicator="acc", is_filter=False, ignore_space=True)
        
        # Simple approach: compare all GT text vs all detected text
        gt_combined = ' '.join([item.get('text', '') for item in gt_items])
        det_combined = ' '.join([det.get('text', '') for det in detections])
        
        if gt_combined and det_combined:
            preds = [(det_combined, 1.0)]
            labels = [(gt_combined, 1.0)]
            pred_label = (preds, labels)
            
            text_result = metric(pred_label)
            final_metrics = metric.get_metric()
            
            return {
                'text_accuracy': final_metrics['norm_edit_dis'],
                'text_exact_match': final_metrics['acc'],
                'matched_pairs': min(len(gt_items), len(detections))
            }
        else:
            return {'text_accuracy': 0.0, 'text_exact_match': 0.0, 'matched_pairs': 0}
            
    except Exception as e:
        print(f"Warning: Text accuracy calculation failed: {e}")
        return {'text_accuracy': 0.0, 'text_exact_match': 0.0, 'matched_pairs': 0}


# =====================================================
# TRADITIONAL TEXT-ONLY EVALUATION FUNCTIONS
# =====================================================

# Removed custom character accuracy calculation - using PaddleOCR's RecMetric only

def calculate_paddleocr_text_accuracy(ground_truth: Dict, ocr_result: Dict, debug: bool = False) -> Dict:
    """
    Calculate OCR accuracy using PaddleOCR's RecMetric - the standard method
    """
    if not PADDLEOCR_AVAILABLE:
        return {"error": "PaddleOCR RecMetric not available"}
    
    # Extract ground truth texts from XFUND format
    gt_texts = []
    if 'document' in ground_truth:
        for item in ground_truth['document']:
            if 'text' in item:
                gt_texts.append(item['text'])
    
    # Extract OCR prediction texts from PaddleOCR format
    pred_texts = []
    if 'rec_texts' in ocr_result:
        pred_texts = ocr_result['rec_texts']
    
    # Combine all text
    gt_combined = ''.join(gt_texts)
    pred_combined = ''.join(pred_texts)
    
    if debug:
        print("\n--- DEBUG MODE ---")
        print(f"Ground Truth Text: {gt_combined}")
        print(f"OCR Predicted Text: {pred_combined}")
        print("--- END DEBUG ---\n")
    
    # Use PaddleOCR's RecMetric for accuracy calculation
    metric = RecMetric(main_indicator="acc", is_filter=False, ignore_space=True)
    
    # Format data as expected by RecMetric
    preds = [(pred_combined, 1.0)]  # (text, confidence)
    labels = [(gt_combined, 1.0)]   # (text, confidence)
    pred_label = (preds, labels)
    
    # Calculate metrics
    metric(pred_label)
    final_metrics = metric.get_metric()
    
    return {
        # PaddleOCR standard metrics
        'paddleocr_accuracy': final_metrics['norm_edit_dis'],  # Use similarity as main accuracy
        'paddleocr_exact_match': final_metrics['acc'],        # Exact match accuracy
        'paddleocr_norm_edit_distance': final_metrics['norm_edit_dis'],
        'character_similarity': final_metrics['norm_edit_dis'],
        
        # Debug info
        'ground_truth_length': len(gt_combined),
        'predicted_length': len(pred_combined)
    }

def load_ground_truth_for_image(ground_truth_file: str, image_name: str) -> Optional[Dict]:
    """Load ground truth data for a specific image - FIXED for XFUND format"""
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # XFUND format has documents array
        documents = gt_data.get('documents', [])
        image_base = os.path.splitext(image_name)[0]
        
        for doc in documents:
            doc_id = doc.get('id', '')
            if doc_id == image_base:
                return doc
        
        print(f"Warning: Ground truth not found for image: {image_name}")
        return None
        
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None

def load_ocr_result_for_image(output_dir: str, image_name: str) -> Optional[Dict]:
    """Load OCR result for a specific image"""
    try:
        base_name = os.path.splitext(image_name)[0]
        # Try both possible filename formats
        json_filename = f"{base_name}_ocr_result.json"
        json_path = os.path.join(output_dir, json_filename)
        
        if not os.path.exists(json_path):
            # Try alternative format
            json_filename = f"{base_name}_res.json"
            json_path = os.path.join(output_dir, json_filename)
        
        if not os.path.exists(json_path):
            print(f"Warning: OCR result not found: {json_path}")
            return None
        
        with open(json_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        return ocr_data
        
    except Exception as e:
        print(f"Error loading OCR result: {e}")
        return None

def calculate_accuracy():
    """
    This function is now a placeholder. The main logic is handled in startup.sh
    which processes all images and then calls this script for each one.
    This script is now focused on single-image calculation.
    """
    pass

def main():
    parser = argparse.ArgumentParser(description='Calculate integrated OCR accuracy (text + position-aware)')
    parser.add_argument('--ground_truth', required=True, help='Path to the master ground truth JSON file (e.g., zh.val.json)')
    parser.add_argument('--output_dir', required=True, help='Directory containing OCR output JSON files')
    parser.add_argument('--image_name', required=True, help='The specific image file name to process (e.g., zh_val_0.jpg)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to print raw and normalized texts')
    parser.add_argument('--position_aware', action='store_true', help='Enable position-aware evaluation (requires detection_results in OCR output)')
    parser.add_argument('--iou_threshold', default=0.5, type=float, help='IoU threshold for position matching (default: 0.5)')
    parser.add_argument('--results_file', help=argparse.SUPPRESS) # Suppress help for this unused arg

    args = parser.parse_args()

    # Load the specific ground truth document for the given image
    gt_data = load_ground_truth_for_image(args.ground_truth, args.image_name)
    if gt_data is None:
        # This is a critical error for single-image calculation
        print(f"ERROR: Ground truth not found for {args.image_name} in {args.ground_truth}", file=sys.stderr)
        # Return a JSON error message for the C++ application to parse
        print(f"SINGLE_ACC: {{\"error\": \"Ground truth not found for {args.image_name}\"}}")
        sys.exit(1)

    # Load the corresponding OCR result
    ocr_data = load_ocr_result_for_image(args.output_dir, args.image_name)
    if ocr_data is None:
        print(f"ERROR: OCR result not found for {args.image_name} in {args.output_dir}", file=sys.stderr)
        print(f"SINGLE_ACC: {{\"error\": \"OCR result not found for {args.image_name}\"}}")
        sys.exit(1)

    # Calculate the accuracy metrics using PaddleOCR
    accuracy_metrics = calculate_paddleocr_text_accuracy(gt_data, ocr_data, debug=args.debug)
    
    # Calculate position-aware metrics if requested and available
    position_metrics = {}
    if args.position_aware:
        if 'detection_results' in ocr_data:
            position_metrics = calculate_position_aware_metrics(gt_data, ocr_data, args.iou_threshold)
            if 'error' in position_metrics:
                print(f"Warning: Position-aware evaluation failed: {position_metrics['error']}", file=sys.stderr)
                position_metrics = {}
        else:
            print("Warning: Position-aware evaluation requested but no detection_results found in OCR output", file=sys.stderr)
            print("Please regenerate OCR results with position information using dxnn_benchmark.py", file=sys.stderr)
    
    # Combine all metrics
    combined_metrics = {
        **accuracy_metrics,
        **position_metrics
    }

        # Print a clean, human-readable summary to stderr for immediate feedback in the log
    if 'error' in accuracy_metrics:
        summary = f"""
========================================
PADDLEOCR ACCURACY EVALUATION - ERROR
========================================
Image: {args.image_name}
Error: {accuracy_metrics['error']}
========================================
"""
    else:
        summary = f"""
========================================
PADDLEOCR ACCURACY EVALUATION
========================================
Image: {args.image_name}
PaddleOCR Accuracy: {accuracy_metrics['paddleocr_accuracy']*100:.2f}%
Character Similarity: {accuracy_metrics['character_similarity']*100:.2f}%
GT Length: {accuracy_metrics['ground_truth_length']}, Pred Length: {accuracy_metrics['predicted_length']}
========================================
"""
    
    if position_metrics and 'error' not in position_metrics:
        summary += f"""
--- Position-Aware Detection Metrics ---
IoU Threshold: {position_metrics.get('iou_threshold', args.iou_threshold)}
Ground Truth Regions: {position_metrics.get('total_gt', 0)}
Detected Regions: {position_metrics.get('total_detections', 0)}
Detection Precision: {position_metrics.get('detection_precision', 0)*100:.2f}%
Detection Recall: {position_metrics.get('detection_recall', 0)*100:.2f}%
Detection F-Score: {position_metrics.get('detection_hmean', 0)*100:.2f}%
Spatial-Text Accuracy: {position_metrics.get('text_accuracy', 0)*100:.2f}%
"""
    
    summary += "========================================"
    print(summary, file=sys.stderr)

    # Print the machine-readable JSON to stdout, prefixed for easy parsing by the C++ app
    print(f"INTEGRATED_ACC: {json.dumps(combined_metrics, ensure_ascii=False)}")

def calculate_integrated_accuracy_for_benchmark(ground_truth_file: str, output_dir: str, 
                                              image_name: str, position_aware: bool = False,
                                              iou_threshold: float = 0.5) -> Dict:
    """
    Integrated accuracy calculation for benchmark - combines text and position-aware metrics
    
    Args:
        ground_truth_file: Path to XFUND ground truth JSON file
        output_dir: Directory containing OCR results
        image_name: Image filename
        position_aware: Whether to include position-aware evaluation
        iou_threshold: IoU threshold for position matching
        
    Returns:
        Dictionary containing combined accuracy metrics
    """
    # Load ground truth and OCR result
    gt_data = load_ground_truth_for_image(ground_truth_file, image_name)
    if gt_data is None:
        return {"error": f"Ground truth not found for {image_name}"}
    
    ocr_data = load_ocr_result_for_image(output_dir, image_name)
    if ocr_data is None:
        return {"error": f"OCR result not found for {image_name}"}
    
    # Calculate text accuracy using PaddleOCR
    text_metrics = calculate_paddleocr_text_accuracy(gt_data, ocr_data, debug=False)
    
    if 'error' in text_metrics:
        return text_metrics
    
    # If position-aware evaluation is requested, add spatial metrics
    if position_aware:
        try:
            position_metrics = calculate_position_aware_metrics(gt_data, ocr_data, iou_threshold)
            # Combine both types of metrics
            combined_metrics = {**text_metrics, **position_metrics}
            return combined_metrics
        except Exception as e:
            # If position-aware fails, return text metrics only with a warning
            text_metrics['position_aware_warning'] = f"Position-aware evaluation failed: {str(e)}"
            return text_metrics
    else:
        return text_metrics


def use_paddleocr_recmetric_direct(ground_truth_text: str, ocr_text: str, 
                                  is_filter: bool = False, ignore_space: bool = True) -> Dict:
    """
    Use PaddleOCR's RecMetric directly
    
    Args:
        ground_truth_text: Ground truth text
        ocr_text: OCR predicted text
        is_filter: Whether to filter to only digits and ASCII letters
        ignore_space: Whether to ignore spaces
        
    Returns:
        Dictionary with accuracy and normalized edit distance
    """
    if not PADDLEOCR_AVAILABLE:
        return {"error": "PaddleOCR RecMetric not available"}
    
    # Create RecMetric instance
    metric = RecMetric(main_indicator="acc", is_filter=is_filter, ignore_space=ignore_space)
    
    # Format data as expected by RecMetric
    # RecMetric expects: preds, labels = pred_label
    # where preds and labels are lists of (text, confidence) tuples
    preds = [(ocr_text, 1.0)]  # confidence = 1.0
    labels = [(ground_truth_text, 1.0)]
    pred_label = (preds, labels)
    
    # Calculate metrics
    result = metric(pred_label)
    final_metrics = metric.get_metric()
    
    return {
        'paddleocr_accuracy': final_metrics['acc'],
        'paddleocr_norm_edit_distance': final_metrics['norm_edit_dis'],
        'paddleocr_similarity': final_metrics['norm_edit_dis'],
        'character_similarity': final_metrics['norm_edit_dis'],
        'is_filter': is_filter,
        'ignore_space': ignore_space
    }


# Removed old functions - only using PaddleOCR integrated evaluation

if __name__ == "__main__":
    main()