# PP-OCRv5 DEEPX Baseline

[中文 README](README_CN.md)

🚀 PP-OCRv5 DEEPX benchmarking toolchain with NPU acceleration and comprehensive performance evaluation.

## 📈 Performance Results

### Custom Dataset Overview

This project uses a diverse custom Chinese dataset for benchmarking. The dataset consists of various real-world scenarios including street signs, handwritten text, exam papers, textbooks, and newspapers, providing comprehensive coverage of different text recognition challenges with detailed annotations including text content and bounding box coordinates.

**Test Configuration**:
- Dataset: Custom Chinese document dataset (20 images)
- Data Format: PNG images with JSON annotations containing text content
- Model: DXNN-OCR v5 full pipeline (PP-OCRv5 → DEEPX NPU accelerated)
  - Text detection: PP-OCRv5 det → DXNN det_v5 (NPU accelerated)
  - Text classification: PP-OCRv5 cls → DXNN cls_v5 (NPU accelerated)
  - Text recognition: PP-OCRv5 rec → DXNN rec_v5 multi-ratio models (NPU accelerated)
- Hardware configuration:
  - Platform: Rockchip RK3588 IR88MX01 LP4X V10
  - NPU: DEEPX DX-M1 Accelerator Card
    - PCIe: Gen3 X4 interface [01:00:00]
    - Firmware: v2.1.0
  - CPU: ARM Cortex-A55 8-core @ 2.35GHz (8nm process)
  - Memory: 8GB LPDDR4X
  - Operating System: Ubuntu 20.04.6 LTS (Focal)
  - Runtime: DXRT v3.0.0 + RT driver v1.7.1 + PCIe driver v1.4.1

**Benchmark Results**:
| NPU Model | Average Inference Time (ms) | Average FPS | Average CPS (chars/s) | Average Accuracy (%) | 
|---|---|---|---|---|
| `DEEPX DX-M1` | 1151.77 | 2.94 | 255.17 | 68.56 |

- [Detailed Performance Results of PP-OCRv5 on DEEPX NPU](./PP-OCRv5_on_DEEEPX.md)

## 🛠️ Quick Start

### ⚡ One Simple Step to Start Your OCR Benchmark

**One-Step Execution:**
```bash
git clone https://github.com/Chris-godz/PP-OCRv5-DeepX-Baseline.git
cd PP-OCRv5-DeepX-Baseline
./startup.sh
```

## 📁 Project Structure

```
├── startup.sh              # One-click benchmark execution
├── scripts/
│   ├── dxnn_benchmark.py   # Main benchmark tool (NPU inference + performance testing)
│   ├── calculate_acc.py    # PP-OCRv5 compatible accuracy calculation
│   └── ocr_engine.py       # DXNN NPU engine interface
├── engine/
│   ├── model_files/v5/     # DXNN v5 NPU models (.dxnn format)
│   ├── draw_utils.py       # Visualization utilities
│   ├── utils.py           # Processing utilities
│   └── fonts/             # Chinese fonts (for visualization)
├── images/                 # Custom dataset (20 PNG images + labels.json)
│   ├── image_1.png ~ image_20.png  # Test images
│   └── labels.json         # Ground truth annotations
├── output/                # Test results output
│   ├── json/              # Detailed JSON results
│   ├── vis/               # Visualization images
│   ├── benchmark_summary.json
│   ├── benchmark_results.csv
│   └── DXNN-OCR_benchmark_report.md
└── logs/                  # Execution logs
```

**Custom Dataset:**
```bash
# Prepare your own images
mkdir -p images/custom
cp /path/to/your/images/* images/custom/

# Run benchmark
python scripts/dxnn_benchmark.py \
    --directory images/custom \
    --ground-truth custom_labels.json \
    --output output_custom \
    --runs 3
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is forked and developed based on [DEEPX-AI/DXNN-OCR](https://github.com/DEEPX-AI/DXNN-OCR) project
- Thanks to [DEEPX team](https://deepx.ai) for NPU runtime and foundational framework support
- Thanks to the [PaddleOCR team](https://github.com/PaddlePaddle/PaddleOCR) for the excellent OCR framework