# DXNN-OCR Benchmark Report

**Test Configuration**:
- Model: PP-OCR v5 (DEEPX NPU acceleration)
- Total Images Tested: 20
- Success Rate: 100.0%

**Test Results**:
| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |
|---|---|---|---|---|
| `image_1.png` | 89.11 | 11.22 | **89.78** | **57.14** |
| `image_10.png` | 1328.12 | 0.75 | **316.99** | **99.75** |
| `image_11.png` | 2861.14 | 0.35 | **332.38** | **99.78** |
| `image_12.png` | 346.63 | 2.88 | **230.80** | **9.77** |
| `image_13.png` | 320.35 | 3.12 | **206.03** | **100.00** |
| `image_14.png` | 1039.08 | 0.96 | **533.16** | **76.01** |
| `image_15.png` | 3696.02 | 0.27 | **409.09** | **98.15** |
| `image_16.png` | 401.05 | 2.49 | **122.18** | **93.75** |
| `image_17.png` | 827.01 | 1.21 | **90.69** | **12.05** |
| `image_18.png` | 1803.54 | 0.55 | **347.65** | **98.48** |
| `image_19.png` | 1053.12 | 0.95 | **364.63** | **49.85** |
| `image_2.png` | 84.16 | 11.88 | **71.29** | **12.00** |
| `image_20.png` | 1678.84 | 0.60 | **181.67** | **53.73** |
| `image_3.png` | 184.52 | 5.42 | **59.61** | **50.00** |
| `image_4.png` | 211.24 | 4.73 | **137.28** | **41.07** |
| `image_5.png` | 115.37 | 8.67 | **173.35** | **95.24** |
| `image_6.png` | 2862.72 | 0.35 | **344.78** | **73.13** |
| `image_7.png` | 830.33 | 1.20 | **351.67** | **69.14** |
| `image_8.png` | 1379.40 | 0.72 | **303.75** | **85.50** |
| `image_9.png` | 1923.65 | 0.52 | **436.67** | **96.75** |
| **Average** | **1151.77** | **2.94** | **255.17** | **68.56** |

**Performance Summary**:
- Average Inference Time: **1151.77 ms**
- Average FPS: **2.94**
- Average CPS: **255.17 chars/s**
- Total Characters Detected: **7636**
- Model Initialization Time: **4629.13 ms**
- Total Processing Time: **72216.61 ms**
- Average Character Accuracy: **68.56%**
- Success Rate: **100.0%** (20/20 images)
