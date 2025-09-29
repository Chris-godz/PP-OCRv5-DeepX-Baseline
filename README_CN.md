# PP-OCRv5 DEEPX 基准测试

[English README](README.md)

🚀 基于 DEEPX NPU 加速的 PP-OCRv5 基准测试工具链，提供全面的性能评估。

## 📈 性能测试结果

### 自定义数据集概述

本项目使用多样化的中文自定义数据集进行基准测试。该数据集包含各种真实场景，包括路标、手写文本、试卷、教科书和报纸，提供不同文本识别挑战的全面覆盖，并包含详细的标注信息（文本内容和边界框坐标）。

**测试配置**:
- 数据集：自定义中文文档数据集（20张图像）
- 数据格式：PNG 图像配合包含文本内容的 JSON 标注
- 模型：DXNN-OCR v5 完整流水线（PP-OCRv5 → DEEPX NPU 加速）
  - 文本检测：PP-OCRv5 det → DXNN det_v5（NPU 加速）
  - 文本分类：PP-OCRv5 cls → DXNN cls_v5（NPU 加速）
  - 文本识别：PP-OCRv5 rec → DXNN rec_v5 多比例模型（NPU 加速）
- 硬件配置：
  - 平台：Rockchip RK3588 IR88MX01 LP4X V10
  - NPU：DEEPX DX-M1 加速卡
    - PCIe：Gen3 X4 接口 [01:00:00]
    - 固件版本：v2.1.0
  - CPU：ARM Cortex-A55 8核心 @ 2.35GHz（8nm 工艺）
  - 系统内存：8GB LPDDR4X
  - 操作系统：Ubuntu 20.04.6 LTS (Focal)
  - 运行时：DXRT v3.0.0 + RT 驱动 v1.7.1 + PCIe 驱动 v1.4.1

**基准测试结果**:
| NPU 型号 | 平均推理时间 (ms) | 平均 FPS | 平均 CPS (字符/秒) | 平均准确率 (%) | 
|---|---|---|---|---|
| `DEEPX DX-M1` | 1151.77 | 2.94 | 255.17 | 68.56 |

- [PP-OCRv5 在 DEEPX NPU 上的详细性能结果](./PP-OCRv5_on_DEEEPX.md)

## 🛠️ 快速开始

### ⚡ 一步启动 OCR 基准测试

**一键执行:**
```bash
git clone https://github.com/Chris-godz/PP-OCRv5-DeepX-Baseline.git
cd PP-OCRv5-DeepX-Baseline
./startup.sh
```

## 📁 项目结构

```
├── startup.sh              # 一键基准测试执行
├── scripts/
│   ├── dxnn_benchmark.py   # 主要基准测试工具（NPU 推理 + 性能测试）
│   ├── calculate_acc.py    # PP-OCRv5 兼容的准确率计算
│   └── ocr_engine.py       # DXNN NPU 引擎接口
├── engine/
│   ├── model_files/v5/     # DXNN v5 NPU 模型（.dxnn 格式）
│   ├── draw_utils.py       # 可视化工具
│   ├── utils.py           # 处理工具
│   └── fonts/             # 中文字体（用于可视化）
├── images/                 # 自定义数据集（20张 PNG 图像 + labels.json）
│   ├── image_1.png ~ image_20.png  # 测试图像
│   └── labels.json         # 真实标注
├── output/                # 测试结果输出
│   ├── json/              # 详细 JSON 结果
│   ├── vis/               # 可视化图像
│   ├── benchmark_summary.json
│   ├── benchmark_results.csv
│   └── DXNN-OCR_benchmark_report.md
└── logs/                  # 执行日志
```

**自定义数据集:**
```bash
# 准备你的图像
mkdir -p images/custom
cp /path/to/your/images/* images/custom/

# 运行基准测试
python scripts/dxnn_benchmark.py \
    --directory images/custom \
    --ground-truth custom_labels.json \
    --output output_custom \
    --runs 3
```

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目基于 [DEEPX-AI/DXNN-OCR](https://github.com/DEEPX-AI/DXNN-OCR) 项目 fork 开发
- 感谢 [DEEPX 团队](https://deepx.ai)提供 NPU 运行时和基础框架支持
- 感谢 [PaddleOCR 团队](https://github.com/PaddlePaddle/PaddleOCR) 提供优秀的 OCR 框架
