"""
OCR Engine utilities and model management
Contains functions for creating and managing OCR engines
"""

from dx_engine import InferenceEngine as IE
from dx_engine import InferenceOption as IO


def make_rec_engines(model_dirname):
    """
    Create recognition engines for different aspect ratios and heights
    
    Args:
        model_dirname (str): Directory containing recognition model files
        
    Returns:
        dict: Dictionary mapping aspect ratios to height-based model dictionaries
        
    Note:
        Creates models for aspect ratios from 5 to 25 in intervals of 10,
        and heights of 10, 20, and 30 pixels
    """
    prefix = ''
    if model_dirname.find('v5') != -1:
        prefix = '_v5'
    
    io = IO().set_use_ort(True)
    rec_model_map = {}
    ratio_interval = 10
    max_ratio = 30

    for i in range(ratio_interval // 2, max_ratio, ratio_interval):
        rec_model_map[i] = {}
        for height in [10, 20, 30]:
            model_path = f"{model_dirname}/rec{prefix}_ratio_{i}_height_{height}.dxnn"
            rec_model_map[i][height] = IE(model_path, io)
    
    return rec_model_map


def create_ocr_models(version='v5'):
    """
    Create detection, classification, and recognition models for v5
    
    Args:
        version (str): Model version (only 'v5' supported)
        
    Returns:
        tuple: (det_model, cls_model, rec_models)
            - det_model: Detection model
            - cls_model: Classification model  
            - rec_models: Dictionary of recognition models
    """
    if version != 'v5':
        raise ValueError(f"Only v5 model is supported, got: {version}")
    
    det_model_path = "engine/model_files/v5/det_v5.dxnn"
    cls_model_path = "engine/model_files/v5/cls_v5.dxnn"
    rec_model_dirname = "engine/model_files/v5/"
    
    det_model = IE(det_model_path, IO().set_use_ort(True))
    cls_model = IE(cls_model_path, IO().set_use_ort(True))
    rec_models = make_rec_engines(rec_model_dirname)
    
    return det_model, cls_model, rec_models


def create_ocr_workers(version='v5', num_workers=3):
    """
    Create multiple OCR worker instances for parallel processing
    
    Args:
        version (str): Model version (only 'v5' supported)
        num_workers (int): Number of worker instances to create
        
    Returns:
        list: List of PaddleOcr worker instances
    """
    from engine.paddleocr import PaddleOcr
    
    det_model, cls_model, rec_models = create_ocr_models(version)
    ocr_workers = [PaddleOcr(det_model, cls_model, rec_models, version) for _ in range(num_workers)]
    
    return ocr_workers 