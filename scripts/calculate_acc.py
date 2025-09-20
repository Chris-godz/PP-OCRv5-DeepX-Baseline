#!/usr/bin/env python3
"""
Research-standard OCR accuracy calculator following academic papers
Implements standard CER and WER calculations used in OCR research
Adapted from PP-OCRv5-Cpp-Baseline for DXNN-OCR project
"""

import json
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional
import unicodedata

def normalize_text_research_standard(text: str) -> str:
    """
    This normalization is designed to be strict for research-level CER/Accuracy.
    It aims to mirror academic standards for OCR evaluation.
    - Converts full-width characters to half-width.
    - Lowercases all text.
    - Removes a comprehensive set of punctuation, symbols, and all whitespace.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unicode normalization to handle combined characters
    text = unicodedata.normalize('NFKC', text)

    # 2. Lowercase the text
    text = text.lower()

    # 3. Build a comprehensive translation table for removal
    # This is far more efficient than repeated .replace() calls
    
    # Combining all forms of punctuation and symbols to be removed.
    # Includes CJK punctuation, general punctuation, and ASCII symbols.
    punctuation_to_remove = "＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～" \
                            "·｜「」『』《》〈〉（）" \
                            ".,;:!?\"'()[]{}<>@#$%^&*-_=+|\\`~" \
                            "●"
    
    # All whitespace characters (space, tab, newline, return, formfeed, vertical tab)
    whitespace_to_remove = " \t\n\r\f\v"

    # The translation table maps the ordinal value of each character to None (for deletion)
    translator = str.maketrans('', '', punctuation_to_remove + whitespace_to_remove)
    
    return text.translate(translator)

def calculate_character_error_rate(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate Character Error Rate (CER) and other metrics using Levenshtein distance.
    """
    ref_norm = normalize_text_research_standard(reference)
    hyp_norm = normalize_text_research_standard(hypothesis)

    if len(ref_norm) == 0:
        return {
            'cer': 0.0 if len(hyp_norm) == 0 else 1.0,
            'accuracy': 1.0 if len(hyp_norm) == 0 else 0.0,
            'substitutions': 0,
            'insertions': len(hyp_norm),
            'deletions': 0,
            'ref_length': 0,
            'hyp_length': len(hyp_norm)
        }

    # Calculate Levenshtein distance and detailed operations
    def levenshtein_with_ops(s1, s2):
        """Calculate Levenshtein distance with operation counts."""
        len1, len2 = len(s1), len(s2)
        
        # Create matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize first row and column
        for i in range(len1 + 1):
            matrix[i][0] = i  # deletions
        for j in range(len2 + 1):
            matrix[0][j] = j  # insertions
        
        # Fill matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1
                
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )
        
        return matrix[len1][len2]
    
    distance = levenshtein_with_ops(ref_norm, hyp_norm)
    error_rate = distance / len(ref_norm)
    accuracy = max(0.0, 1.0 - error_rate)
    
    return {
        'cer': error_rate,
        'accuracy': accuracy,
        'substitutions': distance,  # Simplified - actual substitution count would need backtracking
        'insertions': 0,
        'deletions': 0,
        'ref_length': len(ref_norm),
        'hyp_length': len(hyp_norm)
    }

def calculate_research_standard_accuracy(ground_truth: Dict, ocr_result: Dict, debug: bool = False) -> Dict:
    """
    Calculate OCR accuracy using research-standard methods - FIXED for XFUND format
    """
    
    # Extract ground truth texts from XFUND format - CORRECTED
    gt_texts = []
    if 'document' in ground_truth:
        for item in ground_truth['document']:
            if 'text' in item:
                gt_texts.append(item['text'])
    
    # Extract OCR prediction texts from DXNN-OCR format
    pred_texts = []
    if 'rec_texts' in ocr_result:
        pred_texts = ocr_result['rec_texts']
    elif 'ocr_text' in ocr_result:
        # Handle single string format
        pred_texts = [ocr_result['ocr_text']]
    
    # Combine all text
    gt_combined = ''.join(gt_texts)  # NO SPACES - direct concatenation
    pred_combined = ''.join(pred_texts)  # NO SPACES - direct concatenation
    
    if debug:
        print("\n--- DEBUG MODE ---")
        print(f"RAW Ground Truth:\n---\n{gt_combined}\n---")
        print(f"RAW Prediction:\n---\n{pred_combined}\n---")
        
        gt_norm_debug = normalize_text_research_standard(gt_combined)
        pred_norm_debug = normalize_text_research_standard(pred_combined)
        
        print(f"NORMALIZED Ground Truth:\n---\n{gt_norm_debug}\n---")
        print(f"NORMALIZED Prediction:\n---\n{pred_norm_debug}\n---")
        print("--- END DEBUG ---\n")
    
    # Calculate character-level metrics
    char_metrics = calculate_character_error_rate(gt_combined, pred_combined)
    
    return {
        # ONLY Character Accuracy - simplified output
        'character_accuracy': char_metrics['accuracy'],
        'character_error_rate': char_metrics['cer'],
        
        # Debug info
        'reference_length': char_metrics['ref_length'],
        'hypothesis_length': char_metrics['hyp_length'],
        'edit_distance': char_metrics['substitutions']
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
    """Load OCR result for a specific image from DXNN-OCR output"""
    try:
        base_name = os.path.splitext(image_name)[0]
        json_filename = f"{base_name}_ocr_result.json"
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

def calculate_accuracy_for_dxnn_benchmark(ground_truth_text: str, ocr_text: str, debug: bool = False) -> Dict:
    """
    Calculate accuracy metrics for DXNN benchmark - simplified interface
    """
    # Create mock structures to reuse existing calculation logic
    mock_gt = {'document': [{'text': ground_truth_text}]}
    mock_ocr = {'ocr_text': ocr_text}
    
    return calculate_research_standard_accuracy(mock_gt, mock_ocr, debug)

def main():
    parser = argparse.ArgumentParser(description='Calculate OCR accuracy for DXNN-OCR benchmark')
    parser.add_argument('--ground_truth', required=True, help='Path to the XFUND ground truth JSON file (e.g., zh.val.json)')
    parser.add_argument('--output_dir', required=True, help='Directory containing DXNN-OCR output JSON files')
    parser.add_argument('--image_name', required=True, help='The specific image file name to process (e.g., zh_val_0.jpg)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to print raw and normalized texts')

    args = parser.parse_args()

    # Load the specific ground truth document for the given image
    gt_data = load_ground_truth_for_image(args.ground_truth, args.image_name)
    if gt_data is None:
        print(f"ERROR: Ground truth not found for {args.image_name} in {args.ground_truth}", file=sys.stderr)
        print(f"ACCURACY_RESULT: {{\"error\": \"Ground truth not found for {args.image_name}\"}}")
        sys.exit(1)

    # Load the corresponding OCR result
    ocr_data = load_ocr_result_for_image(args.output_dir, args.image_name)
    if ocr_data is None:
        print(f"ERROR: OCR result not found for {args.image_name} in {args.output_dir}", file=sys.stderr)
        print(f"ACCURACY_RESULT: {{\"error\": \"OCR result not found for {args.image_name}\"}}")
        sys.exit(1)

    # Calculate the accuracy metrics
    accuracy_metrics = calculate_research_standard_accuracy(gt_data, ocr_data, debug=args.debug)

    # Print a clean, human-readable summary
    summary = f"""
========================================
DXNN-OCR ACCURACY EVALUATION
========================================
Image: {args.image_name}
Character Accuracy: {accuracy_metrics['character_accuracy']*100:.2f}%
Character Error Rate: {accuracy_metrics['character_error_rate']*100:.2f}%
Ref Length: {accuracy_metrics['reference_length']}, Hyp Length: {accuracy_metrics['hypothesis_length']}
========================================
"""
    print(summary, file=sys.stderr)

    # Print the machine-readable JSON result
    print(f"ACCURACY_RESULT: {json.dumps(accuracy_metrics, ensure_ascii=False)}")

if __name__ == "__main__":
    main()