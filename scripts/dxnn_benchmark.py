#!/usr/bin/env python3
"""
DXNN-OCR Benchmark Tool
Comprehensive performance evaluation for DXNN OCR engine following PP-OCRv5-Cpp-Baseline methodology
"""

import os
import sys
import json
import time

# Force unbuffered output for real-time printing
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ocr_engine import create_ocr_workers
from scripts.calculate_acc import calculate_integrated_accuracy_for_benchmark


class OCRBenchmark:
    """DXNN OCR Benchmark Tool"""
    
    def __init__(self, version='v5', workers=1, runs_per_image=3, position_aware=False, iou_threshold=0.5):
        """
        Initialize benchmark tool
        
        Args:
            version: Model version (v5)
            workers: Number of worker threads
            runs_per_image: Number of inference runs per image for averaging
            position_aware: Enable position-aware evaluation
            iou_threshold: IoU threshold for position-aware evaluation
        """
        self.version = version
        self.runs_per_image = runs_per_image
        self.position_aware = position_aware
        self.iou_threshold = iou_threshold
        self.ground_truth_file = None  # Will be set when running with ground truth
        self.output_dir = None  # Will be set when saving results
        self.results = []
        
        print(f"[INIT] Initializing DXNN-OCR benchmark with version {version}...")
        start_time = time.time()
        
        try:
            self.ocr_workers = create_ocr_workers(version=version, num_workers=workers)
            self.ocr_engine = self.ocr_workers[0]  # Use first worker
            init_time = time.time() - start_time
            print(f"✓ OCR engine initialized successfully in {init_time*1000:.2f} ms")
            self.init_time_ms = init_time * 1000
        except Exception as e:
            print(f"✗ Failed to initialize OCR engine: {e}")
            sys.exit(1)
    
    def process_single_image(self, image_path: str, ground_truth: Optional[str] = None) -> Dict:
        """
        Process single image with multiple runs for averaging
        
        Args:
            image_path: Path to image file
            ground_truth: Optional ground truth text for accuracy calculation
            
        Returns:
            Dictionary containing benchmark results
        """
        filename = os.path.basename(image_path)
        print(f"[PROCESS] Processing {filename}...", flush=True)
        
        inference_times = []
        total_chars = 0
        ocr_text = ""
        # Store detection results for visualization (from first run only)
        detection_boxes = None
        detection_texts = None
        detection_scores = None
        original_image = None
        
        try:
            # Run multiple inferences for averaging
            print(f"  [INFERENCE] Running {self.runs_per_image} iterations for average metrics...", flush=True)
            
            for run in range(self.runs_per_image):
                print(f"    [RUN {run+1}/{self.runs_per_image}] Starting inference...", flush=True)
                
                # Load image
                import cv2
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                
                # Save original image for visualization (first run only)
                if run == 0:
                    original_image = image.copy()
                
                start_time = time.time()
                boxes, crops, rec_results = self.ocr_engine(image)
                end_time = time.time()
                
                inference_ms = (end_time - start_time) * 1000
                inference_times.append(inference_ms)
                
                # Extract text and character count from first run, save detection results
                if run == 0:
                    texts = []
                    scores = []
                    valid_boxes = []
                    
                    # Process each detection result
                    for i, result_group in enumerate(rec_results):
                        if result_group and i < len(boxes):
                            for text, confidence in result_group:
                                if confidence > 0.3:  # Confidence threshold
                                    texts.append(text)
                                    scores.append(confidence)
                                    valid_boxes.append(boxes[i])
                    
                    ocr_text = ' '.join(texts)
                    # Store for visualization
                    detection_boxes = valid_boxes
                    detection_texts = texts
                    detection_scores = scores
                    total_chars = len(''.join(texts))
                
                print(f"    [RUN {run+1}/{self.runs_per_image}] Completed in {inference_ms:.2f} ms", flush=True)
            
            # Calculate average metrics
            avg_inference_ms = statistics.mean(inference_times)
            min_inference_ms = min(inference_times)
            max_inference_ms = max(inference_times)
            std_inference_ms = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
            
            fps = 1000.0 / avg_inference_ms if avg_inference_ms > 0 else 0.0
            chars_per_second = (total_chars * 1000.0) / avg_inference_ms if avg_inference_ms > 0 else 0.0
            
            # Save OCR result to JSON file for accuracy evaluation
            if self.output_dir:
                try:
                    base_name = os.path.splitext(filename)[0]
                    ocr_result_file = os.path.join(self.output_dir, f"{base_name}_ocr_result.json")
                    
                    ocr_data = {
                        'filename': filename,
                        'ocr_text': ocr_text,
                        'total_chars': total_chars,
                        'avg_inference_ms': avg_inference_ms,
                        'detection_results': []
                    }
                    
                    # Add structured results with bounding boxes and texts if available
                    if detection_boxes is not None and detection_texts is not None:
                        for i, box in enumerate(detection_boxes):
                            detection_item = {
                                'bbox': box.tolist() if hasattr(box, 'tolist') else box,
                                'text': detection_texts[i] if i < len(detection_texts) else '',
                                'confidence': float(detection_scores[i]) if detection_scores and i < len(detection_scores) else 0.0
                            }
                            ocr_data['detection_results'].append(detection_item)
                    
                    # For compatibility, also provide rec_texts format that PaddleOCR expects
                    if detection_texts:
                        ocr_data['rec_texts'] = detection_texts
                    
                    with open(ocr_result_file, 'w', encoding='utf-8') as f:
                        json.dump(ocr_data, f, ensure_ascii=False, indent=2)
                        
                except Exception as e:
                    print(f"  [WARNING] Failed to save OCR result: {e}")
            
            # Calculate accuracy using PaddleOCR integrated evaluation
            accuracy_metrics = None
            if ground_truth and self.ground_truth_file and self.output_dir:
                try:
                    accuracy_metrics = calculate_integrated_accuracy_for_benchmark(
                        self.ground_truth_file, self.output_dir, filename, 
                        self.position_aware, self.iou_threshold
                    )
                except Exception as e:
                    print(f"  [WARNING] PaddleOCR accuracy calculation failed: {e}")
                    accuracy_metrics = None
            
            result = {
                'filename': filename,
                'image_path': image_path,
                'ocr_text': ocr_text,
                'total_chars': total_chars,
                'inference_times_ms': inference_times,
                'avg_inference_ms': avg_inference_ms,
                'min_inference_ms': min_inference_ms,
                'max_inference_ms': max_inference_ms,
                'std_inference_ms': std_inference_ms,
                'fps': fps,
                'chars_per_second': chars_per_second,
                'accuracy_metrics': accuracy_metrics,
                'detection_boxes': detection_boxes,
                'detection_texts': detection_texts, 
                'detection_scores': detection_scores,
                'original_image': original_image,
                'success': True
            }
            
            print(f"  [METRICS] Average inference time: {avg_inference_ms:.2f} ms")
            print(f"  [METRICS] FPS: {fps:.2f}")
            print(f"  [METRICS] Characters/second: {chars_per_second:.2f}")
            print(f"  [METRICS] Total characters detected: {total_chars}")
            
            if accuracy_metrics and 'error' not in accuracy_metrics:
                print(f"  [ACCURACY] ACC: {accuracy_metrics['paddleocr_accuracy']*100:.2f}%")
                
                # Show position-aware metrics if available
                if 'detection_precision' in accuracy_metrics:
                    print(f"  [PRECISION] Precision: {accuracy_metrics['detection_precision']*100:.2f}%")
                    print(f"  [PRECISION] Recall: {accuracy_metrics['detection_recall']*100:.2f}%")
                    print(f"  [PRECISION] F-Score: {accuracy_metrics['detection_hmean']*100:.2f}%")
            
            return result
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")
            return {
                'filename': filename,
                'image_path': image_path,
                'error': str(e),
                'success': False
            }
    
    def process_batch(self, image_paths: List[str], ground_truths: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            ground_truths: Optional dictionary mapping filenames to ground truth texts
            
        Returns:
            List of benchmark results for each image
        """
        print(f"\n[BATCH] Starting batch processing of {len(image_paths)} images...")
        
        batch_start_time = time.time()
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, image_path in enumerate(image_paths):
            print(f"\n[PROGRESS {i+1}/{len(image_paths)}] Processing: {os.path.basename(image_path)}", flush=True)
            
            # Get ground truth if available
            filename = os.path.basename(image_path)
            ground_truth = ground_truths.get(filename) if ground_truths else None
            
            result = self.process_single_image(image_path, ground_truth)
            results.append(result)
            
            if result['success']:
                successful_count += 1
            else:
                failed_count += 1
            
            # Progress update
            if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
                progress = 100.0 * (i + 1) / len(image_paths)
                print(f"\n[PROGRESS] {i+1}/{len(image_paths)} images processed "
                      f"({progress:.1f}%) - Success: {successful_count}, Failed: {failed_count}")
        
        batch_end_time = time.time()
        batch_duration_ms = (batch_end_time - batch_start_time) * 1000
        
        print(f"\n[BATCH] Batch processing completed in {batch_duration_ms:.2f} ms")
        print(f"[BATCH] Success rate: {successful_count}/{len(image_paths)} ({100.0*successful_count/len(image_paths):.1f}%)")
        
        # Store batch-level metrics
        self.batch_metrics = {
            'total_images': len(image_paths),
            'successful_count': successful_count,
            'failed_count': failed_count,
            'success_rate': successful_count / len(image_paths),
            'batch_duration_ms': batch_duration_ms,
            'init_time_ms': self.init_time_ms
        }
        
        return results
    
    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """
        Generate comprehensive summary report
        
        Args:
            results: List of individual image results
            
        Returns:
            Summary statistics dictionary
        """
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {
                'error': 'No successful results to analyze',
                'model_version': self.version,
                'total_images': len(results),
                'successful_images': 0,
                'failed_images': len(results),
                'success_rate_percent': 0.0
            }
        
        # Calculate performance statistics
        inference_times = [r['avg_inference_ms'] for r in successful_results]
        fps_values = [r['fps'] for r in successful_results]
        cps_values = [r['chars_per_second'] for r in successful_results]
        char_counts = [r['total_chars'] for r in successful_results]
        
        # Calculate accuracy statistics if available
        accuracy_values = []
        similarity_values = []
        for r in successful_results:
            if r.get('accuracy_metrics') and 'error' not in r['accuracy_metrics']:
                accuracy_values.append(r['accuracy_metrics']['paddleocr_accuracy'])
                similarity_values.append(r['accuracy_metrics']['character_similarity'])
        
        summary = {
            'model_version': self.version,
            'total_images': len(results),
            'successful_images': len(successful_results),
            'failed_images': len(results) - len(successful_results),
            'success_rate_percent': len(successful_results) / len(results) * 100,
            
            # Performance metrics
            'performance': {
                'avg_inference_time_ms': statistics.mean(inference_times),
                'min_inference_time_ms': min(inference_times),
                'max_inference_time_ms': max(inference_times),
                'std_inference_time_ms': statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0,
                
                'avg_fps': statistics.mean(fps_values),
                'min_fps': min(fps_values),
                'max_fps': max(fps_values),
                
                'avg_chars_per_second': statistics.mean(cps_values),
                'min_chars_per_second': min(cps_values),
                'max_chars_per_second': max(cps_values),
                
                'total_characters_detected': sum(char_counts),
                'avg_characters_per_image': statistics.mean(char_counts),
            },
            
            # Timing information
            'timing': {
                'init_time_ms': self.init_time_ms,
                'batch_duration_ms': getattr(self, 'batch_metrics', {}).get('batch_duration_ms', 0),
                'total_inference_time_ms': sum(inference_times),
            }
        }
        
        # Add accuracy metrics if available
        if accuracy_values:
            summary['accuracy'] = {
                'avg_paddleocr_accuracy_percent': statistics.mean(accuracy_values) * 100,
                'min_paddleocr_accuracy_percent': min(accuracy_values) * 100,
                'max_paddleocr_accuracy_percent': max(accuracy_values) * 100,
                'avg_character_similarity_percent': statistics.mean(similarity_values) * 100,
                'min_character_similarity_percent': min(similarity_values) * 100,
                'max_character_similarity_percent': max(similarity_values) * 100,
            }
        
        # Add position-aware metrics if available
        precision_values = []
        recall_values = []
        fscore_values = []
        for r in successful_results:
            if r.get('accuracy_metrics') and 'error' not in r['accuracy_metrics']:
                if 'detection_precision' in r['accuracy_metrics']:
                    precision_values.append(r['accuracy_metrics']['detection_precision'])
                    recall_values.append(r['accuracy_metrics']['detection_recall'])
                    fscore_values.append(r['accuracy_metrics']['detection_hmean'])
        
        if precision_values:
            summary['position_metrics'] = {
                'avg_precision_percent': statistics.mean(precision_values) * 100,
                'min_precision_percent': min(precision_values) * 100,
                'max_precision_percent': max(precision_values) * 100,
                'avg_recall_percent': statistics.mean(recall_values) * 100,
                'min_recall_percent': min(recall_values) * 100,
                'max_recall_percent': max(recall_values) * 100,
                'avg_fscore_percent': statistics.mean(fscore_values) * 100,
                'min_fscore_percent': min(fscore_values) * 100,
                'max_fscore_percent': max(fscore_values) * 100,
            }
        
        return summary
    
    def print_summary_report(self, summary: Dict, results: List[Dict] = None):
        """Print formatted summary report in PP-OCRv5-Cpp-Baseline style"""
        print("\n" + "="*100)
        print("DXNN-OCR BENCHMARK RESULTS (PP-OCRv5-Cpp-Baseline Compatible Format)")
        print("="*100)
        
        print(f"Model Version: {summary['model_version']}")
        print(f"Total Images: {summary['total_images']}")
        print(f"Successful: {summary['successful_images']}")
        print(f"Failed: {summary['failed_images']}")
        print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
        
        # Check if we have an error (no successful results)
        if 'error' in summary:
            print(f"\n⚠️  {summary['error']}")
            print("="*100)
            return
        
        # Check if position-aware results exist to determine table columns
        has_position_metrics = False
        if results:
            has_position_metrics = any(
                r.get('position_metrics') and 'error' not in r['position_metrics'] 
                for r in results if r['success']
            )
        
        # Table header will be printed by print_pp_ocrv5_style_results
    
    def print_pp_ocrv5_style_results(self, results: List[Dict], summary: Dict):
        """Print detailed results in PP-OCRv5-Cpp-Baseline table format"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("No successful results to display.")
            return
        
        # Check if position-aware results exist to determine table columns
        has_position_metrics = any(
            r.get('accuracy_metrics') and 'error' not in r['accuracy_metrics']
            and 'detection_precision' in r['accuracy_metrics'] 
            for r in successful_results
        )
        
        print("\n**Test Results**:")
        if has_position_metrics:
            print("| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) | Precision (%) |")
            print("|---|---|---|---|---|---|")
        else:
            print("| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |")
            print("|---|---|---|---|---|")
        
        # Print each image result
        for result in successful_results:
            filename = result['filename']
            inference_time = result['avg_inference_ms']
            fps = result['fps']
            cps = result['chars_per_second']
            
            # Get accuracy if available
            accuracy_str = "N/A"
            if result.get('accuracy_metrics') and 'error' not in result['accuracy_metrics']:
                accuracy = result['accuracy_metrics']['paddleocr_accuracy'] * 100
                accuracy_str = f"**{accuracy:.2f}**"
            
            # Get precision if position-aware metrics available
            precision_str = ""
            if has_position_metrics:
                if result.get('accuracy_metrics') and 'error' not in result['accuracy_metrics'] and 'detection_precision' in result['accuracy_metrics']:
                    precision = result['accuracy_metrics']['detection_precision'] * 100
                    precision_str = f" | **{precision:.2f}**"
                else:
                    precision_str = " | N/A"
            
            # Bold format for CPS to match original style
            print(f"| `{filename}` | {inference_time:.2f} | {fps:.2f} | **{cps:.2f}** | {accuracy_str}{precision_str} |")
        
        # Print average row
        if 'performance' in summary:
            perf = summary['performance']
            avg_accuracy_str = "N/A"
            if 'accuracy' in summary:
                avg_accuracy = summary['accuracy']['avg_paddleocr_accuracy_percent']
                avg_accuracy_str = f"**{avg_accuracy:.2f}**"
            
            # Add precision if available
            avg_precision_str = ""
            if has_position_metrics and 'position_metrics' in summary:
                avg_precision = summary['position_metrics']['avg_precision_percent']
                avg_precision_str = f" | **{avg_precision:.2f}**"
            elif has_position_metrics:
                avg_precision_str = " | N/A"
            
            print(f"| **Average** | **{perf['avg_inference_time_ms']:.2f}** | **{perf['avg_fps']:.2f}** | **{perf['avg_chars_per_second']:.2f}** | {avg_accuracy_str}{avg_precision_str} |")
        
        print()
    
    def print_pp_ocrv5_style_summary(self, summary: Dict):
        """Print summary statistics in PP-OCRv5 style"""
        if 'error' in summary:
            return
        
        perf = summary['performance']
        timing = summary['timing']
        
        print("**Performance Summary**:")
        print(f"- Average Inference Time: **{perf['avg_inference_time_ms']:.2f} ms**")
        print(f"- Average FPS: **{perf['avg_fps']:.2f}**")
        print(f"- Average CPS: **{perf['avg_chars_per_second']:.2f} chars/s**")
        print(f"- Total Characters Detected: **{perf['total_characters_detected']}**")
        print(f"- Model Initialization Time: **{timing['init_time_ms']:.2f} ms**")
        print(f"- Total Processing Time: **{timing['batch_duration_ms']:.2f} ms**")
        
        if 'accuracy' in summary:
            acc = summary['accuracy']
            print(f"- Average PaddleOCR Accuracy: **{acc['avg_paddleocr_accuracy_percent']:.2f}%**")
        
        print(f"- Success Rate: **{summary['success_rate_percent']:.1f}%** ({summary['successful_images']}/{summary['total_images']} images)")
        print()
    
    def save_visualization_results(self, results: List[Dict], output_dir: str):
        """
        Generate and save visualization images for OCR results
        
        Args:
            results: Individual image results containing detection data
            output_dir: Output directory path
        """
        vis_dir = os.path.join(output_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        
        print(f"\n[VIS] Generating visualization images...")
        
        try:
            # Import visualization functions
            from engine.draw_utils import draw_ocr
            import cv2
            
            vis_count = 0
            for result in results:
                if (result['success'] and 
                    result.get('detection_boxes') and 
                    result.get('original_image') is not None):
                    
                    try:
                        filename = result['filename']
                        base_name = os.path.splitext(filename)[0]
                        vis_output_path = os.path.join(vis_dir, f"{base_name}_result.jpg")
                        
                        # Get detection results
                        boxes = result['detection_boxes']
                        texts = result.get('detection_texts', [])
                        scores = result.get('detection_scores', [])
                        original_image = result['original_image']
                        
                        if boxes and len(boxes) > 0:
                            # Generate visualization using engine's draw function
                            vis_image = draw_ocr(
                                image=original_image,
                                boxes=boxes,
                                txts=texts if texts else None,
                                scores=scores if scores else None,
                                drop_score=0.3  # Same threshold as processing
                            )
                            
                            # Save visualization image with high quality
                            cv2.imwrite(vis_output_path, vis_image, [cv2.IMWRITE_JPEG_QUALITY, 98])
                            vis_count += 1
                            print(f"  [VIS] Saved: {base_name}_result.jpg")
                        
                    except Exception as e:
                        print(f"  [VIS] Failed to generate visualization for {filename}: {e}")
                        continue
            
            print(f"[VIS] Generated {vis_count} visualization images in {vis_dir}/")
            
        except ImportError as e:
            print(f"[VIS] Visualization not available: {e}")
        except Exception as e:
            print(f"[VIS] Visualization generation failed: {e}")
    
    def save_results(self, results: List[Dict], summary: Dict, output_dir: str):
        """
        Save detailed results and summary to files
        
        Args:
            results: Individual image results
            summary: Summary statistics
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create PP-OCRv5 style subdirectories
        json_dir = os.path.join(output_dir, 'json')
        vis_dir = os.path.join(output_dir, 'vis')
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save detailed results in json subdirectory (exclude large image data)
        results_file = os.path.join(json_dir, 'benchmark_detailed_results.json')
        
        # Clean results for JSON serialization (remove large data)
        json_safe_results = []
        for result in results:
            clean_result = {k: v for k, v in result.items() 
                          if k not in ['original_image', 'detection_boxes', 'detection_texts', 'detection_scores']}
            json_safe_results.append(clean_result)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results, f, ensure_ascii=False, indent=2)
        
        # Save summary in main output directory (PP-OCRv5 style)
        summary_file = os.path.join(output_dir, 'benchmark_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Save CSV format for easy analysis
        csv_file = os.path.join(output_dir, 'benchmark_results.csv')
        with open(csv_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("filename,avg_inference_ms,fps,chars_per_second,total_chars,paddleocr_accuracy,character_similarity,detection_precision,detection_recall,detection_fscore\n")
            
            # Write data
            for result in results:
                if result['success']:
                    acc_metrics = result.get('accuracy_metrics', {})
                    if acc_metrics and 'error' not in acc_metrics:
                        accuracy = acc_metrics.get('paddleocr_accuracy', '')
                        similarity = acc_metrics.get('character_similarity', '')
                        precision = acc_metrics.get('detection_precision', '')
                        recall = acc_metrics.get('detection_recall', '')
                        fscore = acc_metrics.get('detection_hmean', '')
                    else:
                        accuracy = similarity = precision = recall = fscore = ''
                    
                    f.write(f"{result['filename']},{result['avg_inference_ms']:.2f},"
                           f"{result['fps']:.2f},{result['chars_per_second']:.2f},"
                           f"{result['total_chars']},{accuracy},{similarity},{precision},{recall},{fscore}\n")
        
        # Save PP-OCRv5 style markdown report
        markdown_file = os.path.join(output_dir, 'DXNN-OCR_benchmark_report.md')
        self.save_pp_ocrv5_style_markdown(results, summary, markdown_file)
        
        # Generate visualization images
        self.save_visualization_results(results, output_dir)
        
        print(f"\n[SAVE] Results saved to {output_dir}/")
        print(f"  - Detailed results: json/benchmark_detailed_results.json")
        print(f"  - Summary: benchmark_summary.json") 
        print(f"  - CSV format: benchmark_results.csv")
        print(f"  - PP-OCRv5 style report: DXNN-OCR_benchmark_report.md")
        print(f"  - Visualization images: vis/")
    
    def save_pp_ocrv5_style_markdown(self, results: List[Dict], summary: Dict, output_file: str):
        """Save benchmark results in PP-OCRv5-Cpp-Baseline markdown format"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# DXNN-OCR Benchmark Report\n\n")
            
            # Test Configuration
            f.write("**Test Configuration**:\n")
            f.write(f"- Model: PP-OCR {summary['model_version']} (DEEPX NPU acceleration)\n")
            f.write(f"- Total Images Tested: {summary['total_images']}\n")
            f.write(f"- Success Rate: {summary['success_rate_percent']:.1f}%\n\n")
            
            # Test Results Table - Check if we have position-aware metrics
            has_position_metrics = any(
                r.get('accuracy_metrics', {}).get('detection_precision') is not None 
                for r in successful_results
            )
            
            f.write("**Test Results**:\n")
            if has_position_metrics:
                f.write("| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) | Precision (%) |\n")
                f.write("|---|---|---|---|---|---|\n")
            else:
                f.write("| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |\n")
                f.write("|---|---|---|---|---|\n")
            
            # Write each result
            for result in successful_results:
                filename = result['filename']
                inference_time = result['avg_inference_ms']
                fps = result['fps']
                cps = result['chars_per_second']
                
                accuracy_str = "N/A"
                precision_str = "N/A"
                
                if result.get('accuracy_metrics'):
                    accuracy = result['accuracy_metrics']['paddleocr_accuracy'] * 100
                    accuracy_str = f"**{accuracy:.2f}**"
                    
                    if 'detection_precision' in result['accuracy_metrics']:
                        precision = result['accuracy_metrics']['detection_precision'] * 100
                        precision_str = f"**{precision:.2f}**"
                
                if has_position_metrics:
                    f.write(f"| `{filename}` | {inference_time:.2f} | {fps:.2f} | **{cps:.2f}** | {accuracy_str} | {precision_str} |\n")
                else:
                    f.write(f"| `{filename}` | {inference_time:.2f} | {fps:.2f} | **{cps:.2f}** | {accuracy_str} |\n")
            
            # Average row
            if 'performance' in summary:
                perf = summary['performance']
                avg_accuracy_str = "N/A"
                avg_precision_str = "N/A"
                
                if 'accuracy' in summary:
                    avg_accuracy = summary['accuracy']['avg_paddleocr_accuracy_percent']
                    avg_accuracy_str = f"**{avg_accuracy:.2f}**"
                
                # Calculate average precision if available
                if has_position_metrics:
                    precision_values = []
                    for result in successful_results:
                        acc_metrics = result.get('accuracy_metrics', {})
                        if 'detection_precision' in acc_metrics:
                            precision_values.append(acc_metrics['detection_precision'])
                    
                    if precision_values:
                        avg_precision = sum(precision_values) / len(precision_values) * 100
                        avg_precision_str = f"**{avg_precision:.2f}**"
                    
                    f.write(f"| **Average** | **{perf['avg_inference_time_ms']:.2f}** | **{perf['avg_fps']:.2f}** | **{perf['avg_chars_per_second']:.2f}** | {avg_accuracy_str} | {avg_precision_str} |\n\n")
                else:
                    f.write(f"| **Average** | **{perf['avg_inference_time_ms']:.2f}** | **{perf['avg_fps']:.2f}** | **{perf['avg_chars_per_second']:.2f}** | {avg_accuracy_str} |\n\n")
            
            # Performance Summary
            f.write("**Performance Summary**:\n")
            if 'performance' in summary:
                perf = summary['performance']
                timing = summary['timing']
                
                f.write(f"- Average Inference Time: **{perf['avg_inference_time_ms']:.2f} ms**\n")
                f.write(f"- Average FPS: **{perf['avg_fps']:.2f}**\n")
                f.write(f"- Average CPS: **{perf['avg_chars_per_second']:.2f} chars/s**\n")
                f.write(f"- Total Characters Detected: **{perf['total_characters_detected']}**\n")
                f.write(f"- Model Initialization Time: **{timing['init_time_ms']:.2f} ms**\n")
                f.write(f"- Total Processing Time: **{timing['batch_duration_ms']:.2f} ms**\n")
                
                if 'accuracy' in summary:
                    acc = summary['accuracy']
                    f.write(f"- Average PaddleOCR Accuracy: **{acc['avg_paddleocr_accuracy_percent']:.2f}%**\n")
                
                f.write(f"- Success Rate: **{summary['success_rate_percent']:.1f}%** ({summary['successful_images']}/{summary['total_images']} images)\n")
                
                # Add position-aware metrics to Performance Summary if available
                if has_position_metrics:
                    precision_values = []
                    recall_values = []
                    fscore_values = []
                    
                    for result in successful_results:
                        acc_metrics = result.get('accuracy_metrics', {})
                        if 'detection_precision' in acc_metrics:
                            precision_values.append(acc_metrics['detection_precision'])
                            recall_values.append(acc_metrics['detection_recall'])
                            fscore_values.append(acc_metrics['detection_hmean'])
                    
                    if precision_values:
                        avg_precision = sum(precision_values) / len(precision_values) * 100
                        avg_recall = sum(recall_values) / len(recall_values) * 100
                        avg_fscore = sum(fscore_values) / len(fscore_values) * 100
                        
                        f.write(f"- Average Detection Precision: **{avg_precision:.2f}%**\n")
                        f.write(f"- Average Detection Recall: **{avg_recall:.2f}%**\n")
                        f.write(f"- Average Detection F-Score: **{avg_fscore:.2f}%**\n")


def load_xfund_ground_truth(json_path: str) -> Dict[str, str]:
    """
    Load XFUND dataset ground truth annotations
    
    Args:
        json_path: Path to XFUND JSON annotation file
        
    Returns:
        Dictionary mapping image filenames to ground truth text
    """
    if not os.path.exists(json_path):
        print(f"Warning: Ground truth file not found: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ground_truths = {}
        
        # Handle XFUND format: {'documents': [...], ...}
        documents = data.get('documents', [])
        if not documents and isinstance(data, list):
            documents = data  # Fallback for direct list format
        
        for doc in documents:
            # Get image filename from document
            img_info = doc.get('img', {})
            filename = img_info.get('fname', '')
            
            if filename:
                # Extract all text from document entities
                texts = []
                document_entities = doc.get('document', [])
                for entity in document_entities:
                    text = entity.get('text', '').strip()
                    if text:
                        texts.append(text)
                
                ground_truths[filename] = ' '.join(texts)
        
        print(f"Loaded ground truth for {len(ground_truths)} images from {json_path}")
        return ground_truths
        
    except Exception as e:
        print(f"Error loading ground truth from {json_path}: {e}")
        return {}


def find_image_files(path: str, recursive: bool = False) -> List[str]:
    """Find image files in directory or return single file"""
    if os.path.isfile(path):
        return [path] if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')) else []
    
    if not os.path.isdir(path):
        return []
    
    image_files = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if recursive:
        for root, dirs, files in os.walk(path):
            for file in files:
                if Path(file).suffix.lower() in supported_formats:
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in supported_formats:
                image_files.append(file_path)
    
    return sorted(image_files)


def main():
    """Main entry point for DXNN-OCR benchmark tool"""
    parser = argparse.ArgumentParser(
        description="DXNN-OCR Benchmark Tool - Following PP-OCRv5-Cpp-Baseline methodology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s test_images/1.jpg                                    # Single image
  %(prog)s -d test_images/                                      # Directory
  %(prog)s -d test_images/ --ground-truth xfund/zh.val.json   # With accuracy evaluation
  %(prog)s -d test_images/ --output results/ --runs 5          # Custom settings
        """
    )
    
    # Input options
    parser.add_argument('images', nargs='*', help='Image file paths to process (mutually exclusive with --directory)')
    parser.add_argument('-d', '--directory', help='Directory containing images (mutually exclusive with image files)')
    
    # Processing options
    parser.add_argument('--recursive', action='store_true',
                       help='Process directory recursively')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of inference runs per image for averaging (default: 3)')
    
    # Accuracy evaluation
    parser.add_argument('--ground-truth', help='Path to XFUND ground truth JSON file')
    parser.add_argument('--position-aware', action='store_true', 
                       help='Enable position-aware evaluation (Precision/Recall/F-Score)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for position-aware evaluation (default: 0.5)')
    
    # Output options
    parser.add_argument('--output', '-o', help='Directory to save results')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual OCR results as JSON files')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if args.images and args.directory:
        print("Error: Cannot specify both image files and directory. Use either files or --directory.")
        sys.exit(1)
    
    if not args.images and not args.directory:
        print("Error: Must specify either image file(s) or --directory")
        sys.exit(1)
    
    # Collect input files
    if args.images:
        image_files = []
        for path in args.images:
            if os.path.isfile(path):
                image_files.append(path)
            else:
                print(f"Warning: File not found: {path}")
        
        if not image_files:
            print("Error: No valid image files provided")
            sys.exit(1)
            
    elif args.directory:
        image_files = find_image_files(args.directory, args.recursive)
        if not image_files:
            print(f"Error: No image files found in {args.directory}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} image files")
    
    # Load ground truth if provided
    ground_truths = None
    if args.ground_truth:
        ground_truths = load_xfund_ground_truth(args.ground_truth)
    
    # Initialize benchmark
    benchmark = OCRBenchmark(version='v5', workers=1, runs_per_image=args.runs, 
                           position_aware=args.position_aware, iou_threshold=args.iou_threshold)
    
    # Set ground truth file and output directory for integrated evaluation BEFORE processing
    if args.ground_truth:
        benchmark.ground_truth_file = args.ground_truth
    if args.output:
        # Create output directory structure first
        json_dir = os.path.join(args.output, 'json')
        os.makedirs(json_dir, exist_ok=True)
        benchmark.output_dir = json_dir
    
    # Process images
    results = benchmark.process_batch(image_files, ground_truths)
    
    # Generate and print summary
    summary = benchmark.generate_summary_report(results)
    benchmark.print_summary_report(summary, results)
    benchmark.print_pp_ocrv5_style_results(results, summary)
    benchmark.print_pp_ocrv5_style_summary(summary)
    
    # Save results if output directory specified
    if args.output:
        benchmark.save_results(results, summary, args.output)
        
        # Save individual OCR results if requested
        if args.save_individual:
            json_dir = os.path.join(args.output, 'json')
            for result in results:
                if result['success']:
                    filename = result['filename']
                    base_name = os.path.splitext(filename)[0]
                    ocr_result_file = os.path.join(json_dir, f"{base_name}_ocr_result.json")
                    
                    ocr_data = {
                        'filename': filename,
                        'ocr_text': result['ocr_text'],
                        'total_chars': result['total_chars'],
                        'avg_inference_ms': result['avg_inference_ms'],
                        # Add structured OCR results with position information
                        'detection_results': []
                    }
                    
                    # Add structured results with bounding boxes and texts if available
                    if 'detection_boxes' in result and 'detection_texts' in result:
                        boxes = result['detection_boxes'] 
                        texts = result.get('detection_texts', [])
                        scores = result.get('detection_scores', [])
                        
                        for i, box in enumerate(boxes):
                            detection_item = {
                                'bbox': box.tolist() if hasattr(box, 'tolist') else box,
                                'text': texts[i] if i < len(texts) else '',
                                'confidence': float(scores[i]) if i < len(scores) else 0.0
                            }
                            ocr_data['detection_results'].append(detection_item)
                    
                    # For compatibility, also provide rec_texts format that PaddleOCR expects
                    if 'detection_texts' in result:
                        ocr_data['rec_texts'] = result['detection_texts']
                    
                    with open(ocr_result_file, 'w', encoding='utf-8') as f:
                        json.dump(ocr_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()