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
import unicodedata

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ocr_engine import create_ocr_workers
from scripts.calculate_acc import calculate_accuracy_for_dxnn_benchmark


class OCRBenchmark:
    """DXNN OCR Benchmark Tool"""
    
    def __init__(self, version='v5', workers=1, runs_per_image=3):
        """
        Initialize benchmark tool
        
        Args:
            version: Model version (v5)
            workers: Number of worker threads
            runs_per_image: Number of inference runs per image for averaging
        """
        self.version = version
        self.runs_per_image = runs_per_image
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
    
    def normalize_text_research_standard(self, text: str) -> str:
        """
        Normalize text for research-standard accuracy calculation
        Following PP-OCRv5-Cpp-Baseline methodology
        """
        if not isinstance(text, str):
            return ""

        # Unicode normalization to handle combined characters
        text = unicodedata.normalize('NFKC', text)
        
        # Lowercase the text
        text = text.lower()
        
        # Remove punctuation and whitespace
        punctuation_to_remove = "＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～" \
                               "·｜「」『』《》〈〉（）" \
                               ".,;:!?\"'()[]{}<>@#$%^&*-_=+|\\`~" \
                               "●"
        
        whitespace_to_remove = " \t\n\r\f\v"
        translator = str.maketrans('', '', punctuation_to_remove + whitespace_to_remove)
        
        return text.translate(translator)
    
    def calculate_character_accuracy(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate character-level accuracy metrics
        
        Args:
            reference: Ground truth text
            hypothesis: OCR predicted text
            
        Returns:
            Dictionary containing accuracy metrics
        """
        # Normalize both texts
        ref_norm = self.normalize_text_research_standard(reference)
        hyp_norm = self.normalize_text_research_standard(hypothesis)
        
        if len(ref_norm) == 0:
            return {
                'character_accuracy': 1.0 if len(hyp_norm) == 0 else 0.0,
                'character_error_rate': 0.0 if len(hyp_norm) == 0 else 1.0,
                'reference_length': 0,
                'hypothesis_length': len(hyp_norm)
            }
        
        # Calculate Levenshtein distance
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        edit_distance = levenshtein_distance(ref_norm, hyp_norm)
        cer = edit_distance / len(ref_norm)
        accuracy = 1.0 - cer
        
        return {
            'character_accuracy': max(0.0, accuracy),
            'character_error_rate': cer,
            'reference_length': len(ref_norm),
            'hypothesis_length': len(hyp_norm),
            'edit_distance': edit_distance
        }
    
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
            
            # Calculate accuracy if ground truth is provided
            accuracy_metrics = None
            if ground_truth:
                try:
                    accuracy_metrics = calculate_accuracy_for_dxnn_benchmark(ground_truth, ocr_text)
                except Exception as e:
                    print(f"  [WARNING] Accuracy calculation failed: {e}")
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
            
            if accuracy_metrics:
                print(f"  [ACCURACY] Character accuracy: {accuracy_metrics['character_accuracy']*100:.2f}%")
                print(f"  [ACCURACY] Character error rate: {accuracy_metrics['character_error_rate']*100:.2f}%")
            
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
        cer_values = []
        for r in successful_results:
            if r.get('accuracy_metrics'):
                accuracy_values.append(r['accuracy_metrics']['character_accuracy'])
                cer_values.append(r['accuracy_metrics']['character_error_rate'])
        
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
                'avg_character_accuracy_percent': statistics.mean(accuracy_values) * 100,
                'min_character_accuracy_percent': min(accuracy_values) * 100,
                'max_character_accuracy_percent': max(accuracy_values) * 100,
                'avg_character_error_rate_percent': statistics.mean(cer_values) * 100,
                'min_character_error_rate_percent': min(cer_values) * 100,
                'max_character_error_rate_percent': max(cer_values) * 100,
            }
        
        return summary
    
    def print_summary_report(self, summary: Dict):
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
        
        print("\n**Test Results**:")
        print("| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |")
        print("|---|---|---|---|---|")
        
        print("="*100)
    
    def print_pp_ocrv5_style_results(self, results: List[Dict], summary: Dict):
        """Print detailed results in PP-OCRv5-Cpp-Baseline table format"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("No successful results to display.")
            return
        
        print("\n**Test Results**:")
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
            if result.get('accuracy_metrics'):
                accuracy = result['accuracy_metrics']['character_accuracy'] * 100
                accuracy_str = f"**{accuracy:.2f}**"
            
            # Bold format for CPS to match original style
            print(f"| `{filename}` | {inference_time:.2f} | {fps:.2f} | **{cps:.2f}** | {accuracy_str} |")
        
        # Print average row
        if 'performance' in summary:
            perf = summary['performance']
            avg_accuracy_str = "N/A"
            if 'accuracy' in summary:
                avg_accuracy = summary['accuracy']['avg_character_accuracy_percent']
                avg_accuracy_str = f"**{avg_accuracy:.2f}**"
            
            print(f"| **Average** | **{perf['avg_inference_time_ms']:.2f}** | **{perf['avg_fps']:.2f}** | **{perf['avg_chars_per_second']:.2f}** | {avg_accuracy_str} |")
        
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
            print(f"- Average Character Accuracy: **{acc['avg_character_accuracy_percent']:.2f}%**")
        
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
                            
                            # Save visualization image
                            cv2.imwrite(vis_output_path, vis_image)
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
            f.write("filename,avg_inference_ms,fps,chars_per_second,total_chars,character_accuracy,character_error_rate\n")
            
            # Write data
            for result in results:
                if result['success']:
                    acc_metrics = result.get('accuracy_metrics', {})
                    accuracy = acc_metrics.get('character_accuracy', '') if acc_metrics else ''
                    cer = acc_metrics.get('character_error_rate', '') if acc_metrics else ''
                    
                    f.write(f"{result['filename']},{result['avg_inference_ms']:.2f},"
                           f"{result['fps']:.2f},{result['chars_per_second']:.2f},"
                           f"{result['total_chars']},{accuracy},{cer}\n")
        
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
            
            # Test Results Table
            f.write("**Test Results**:\n")
            f.write("| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |\n")
            f.write("|---|---|---|---|---|\n")
            
            # Write each result
            for result in successful_results:
                filename = result['filename']
                inference_time = result['avg_inference_ms']
                fps = result['fps']
                cps = result['chars_per_second']
                
                accuracy_str = "N/A"
                if result.get('accuracy_metrics'):
                    accuracy = result['accuracy_metrics']['character_accuracy'] * 100
                    accuracy_str = f"**{accuracy:.2f}**"
                
                f.write(f"| `{filename}` | {inference_time:.2f} | {fps:.2f} | **{cps:.2f}** | {accuracy_str} |\n")
            
            # Average row
            if 'performance' in summary:
                perf = summary['performance']
                avg_accuracy_str = "N/A"
                if 'accuracy' in summary:
                    avg_accuracy = summary['accuracy']['avg_character_accuracy_percent']
                    avg_accuracy_str = f"**{avg_accuracy:.2f}**"
                
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
                    f.write(f"- Average Character Accuracy: **{acc['avg_character_accuracy_percent']:.2f}%**\n")
                
                f.write(f"- Success Rate: **{summary['success_rate_percent']:.1f}%** ({summary['successful_images']}/{summary['total_images']} images)\n")


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
    benchmark = OCRBenchmark(version='v5', workers=1, runs_per_image=args.runs)
    
    # Process images
    results = benchmark.process_batch(image_files, ground_truths)
    
    # Generate and print summary
    summary = benchmark.generate_summary_report(results)
    benchmark.print_summary_report(summary)
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
                        'avg_inference_ms': result['avg_inference_ms']
                    }
                    
                    with open(ocr_result_file, 'w', encoding='utf-8') as f:
                        json.dump(ocr_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()