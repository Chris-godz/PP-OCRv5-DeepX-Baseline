# PP-OCRv5 DEEEPX Benchmark

[中文 README](README_CN.md)

🚀 PP-OCRv5 benchmarking toolchain with DEEPX NPU acceleration and comprehensive performance evaluation.

## 📈 Performance Results

### XFUND Dataset Overview

This project uses the [XFUND](https://github.com/doc-analysis/XFUND) dataset for benchmarking. XFUND (eXtended FUnctional Needs Dataset) is a large-scale multilingual form understanding dataset released by Microsoft, containing form images and structured annotations in 7 languages (Chinese, English, Japanese, Spanish, French, Italian, German).

**Test Configuration**:
- Dataset: XFUND Chinese validation set (50 images)
- Model: DXNN-OCR v5 full pipeline (based on PP-OCRv5, DEEPX NPU accelerated)
  - Text detection: PP-OCRv5 det → DXNN det_v5 (NPU optimized)
  - Text classification: PP-OCRv5 cls → DXNN cls_v5 (NPU optimized)
  - Text recognition: PP-OCRv5 rec → DXNN rec_v5 multi-ratio models (NPU optimized)
- Hardware configuration:
  - Platform: Rockchip RK3588 IR88MX01 LP4X V10
  - NPU: DEEPX DX-M1 Accelerator Card
    - PCIe: Gen3 X4 interface [01:00:00]
    - Firmware: v2.1.0
  - CPU: ARM Cortex-A55 8-core @ 2.35GHz (8nm process)
  - System Memory: 8GB LPDDR4X
  - Operating System: Ubuntu 20.04.6 LTS (Focal)
  - Runtime: DXRT v2.9.5 + RT driver v1.7.1 + PCIe driver v1.4.1

**Benchmark Results**:
| Hardware | Average Inference Time (ms) | Average FPS | Average CPS (chars/s) | Average Accuracy (%) | 
|---|---|---|---|---|
| `DEEPX NPU` | 1767.31 | 0.64 | 250.57 | 46.41 |

For detailed test results, see: [PP-OCRv5_on_DEEEPX.md](PP-OCRv5_on_DEEEPX.md)

## 🛠️ Quick Start

### ⚡ One Simple Step to Start Your OCR Benchmark

**One-Step Execution:**
```bash
git clone https://github.com/DEEPX-AI/DXNN-OCR.git
cd DXNN-OCR
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
│   ├── model_files/v5/     # DXNN v5 NPU models
│   ├── draw_utils.py       # Visualization utilities
│   ├── utils.py           # Processing utilities
│   └── fonts/             # Chinese fonts (for visualization)
├── images/xfund/          # XFUND dataset (auto-downloaded)
├── output/                # PP-OCRv5 compatible results
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
    --output output_custom \
    --runs 3
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is forked and developed based on [DEEPX-AI/DXNN-OCR](https://github.com/DEEPX-AI/DXNN-OCR) project
- Thanks to [DEEPX team](https://deepx.ai) for NPU runtime and foundational framework support
- Thanks to the [PaddleOCR team](https://github.com/PaddlePaddle/PaddleOCR) for the excellent OCR framework
- Thanks to [XFUND dataset](https://github.com/doc-analysis/XFUND) for providing standardized evaluation data 
