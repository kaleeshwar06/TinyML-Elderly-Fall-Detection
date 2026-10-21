# TinyML Context-Aware Elderly Fall Detection & Live Testing Suite

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![TinyML](https://img.shields.io/badge/Embedded-ESP32%20%7C%20FreeRTOS-orange.svg)](#hardware-and-embedded-implementation)
[![Model](https://img.shields.io/badge/Model-TinyFallNet%20(SE--ResCNN)-green.svg)](#proposed-tinyfallnet-architecture)
[![Evaluation](https://img.shields.io/badge/Dataset-SisFall%20(92%2C703%20Windows)-blueviolet.svg)](#dataset--benchmark-evaluation)

An end-to-end TinyML and edge-simulation framework for real-time elderly fall detection, pre-fall loss-of-balance mitigation, and kinematic severity stratification using 6-axis IMU inertial sensors (accelerometer and gyroscope).

---

## 👥 Team & Individual Responsibilities

| Contributor | Specific Role & Project Module |
| :--- | :--- |
| **Bharath Palanisamy Mettukadai** | **Hardware Implementation & Sensor Interfacing**<br>ESP32 dual-core microcontroller setup, MPU-6050 6-axis IMU I2C communication (GPIO 21/22, 0x68/0x70 raw registers), calibrated $g$ and $\text{deg/s}$ physical conversions, and resultant acceleration magnitude ($\|a\|$) calculation. |
| **Kaleeshwar** | **Machine Learning & Edge Optimization (TinyFallNet)**<br>SisFall benchmark data modeling (92,703 temporal windows), Squeeze-and-Excitation Residual CNN (SE-ResCNN) architecture design, Class-Weighted Focal Loss for extreme class imbalance ($\sim 96.2\%$ ADL vs. $2.95\%$ Fall vs. $0.85\%$ Pre-Fall), PyTorch model training, INT8 quantization, and hardware profiling (Pareto frontier). |
| **Hiba Fathima** | **Embedded Coding & ML Testing / Publication UI**<br>FreeRTOS test firmware pipeline, threshold-sweep evaluation framework, interactive real-time simulation suites ([`dashboard.html`](dashboard.html) and [`live_testing.html`](live_testing.html)), comprehensive documentation, and literature synthesis. |

---

## 🚀 Key Features

1. **Proposed TinyFallNet Architecture (SE-ResCNN):**
   - Temporal depthwise separable convolutions with squeeze-and-excitation channel attention.
   - Ultra-low memory footprint: **28.4 KB Flash**, **11.2 KB RAM**, and **12.4 ms inference latency** on 240 MHz Tensilica Xtensa LX6 (ESP32).
   - Macro F1-score of **96.8%** and Fall Sensitivity of **98.4%** at calibrated threshold $\tau = 0.50$.
2. **Context-Aware Multi-Stage Classification:**
   - **Class 0 (ADL / Normal Activity):** Everyday motions (walking, sitting, standing, stairs).
   - **Class 1 (Pre-Fall / Loss of Balance):** Stumbles, trips, and postural instability before ground impact.
   - **Class 2 (Confirmed Fall Impact):** High-kinetic acceleration impact followed by quiescence.
3. **Kinematic Severity Index (SI):**
   - Instantaneous severity grading combining peak acceleration, rotational velocity, and post-impact immobility variance:
     $$\text{SI} = 50 \cdot \min\left(\frac{\|a\|_{\text{peak}}}{3}, 1\right) + 30 \cdot \min\left(\frac{\|\omega\|_{\text{peak}}}{5}, 1\right) + 20 \cdot \left(\frac{1}{1 + 50 \cdot \text{Var}(a_{\text{post}})}\right)$$
4. **Interactive Browser-Based Testing & Visualization Suite:**
   - [`live_testing.html`](live_testing.html): Standalone, zero-latency testing interface with real-time sliders, presets (Walking, Sitting, Trip, Slip, Syncope), dynamic 256-sample IMU waveform generator, and softmax probability distributions.
   - [`dashboard.html`](dashboard.html): Comprehensive 9-section academic review dashboard with publication-quality charts and ablation summaries.

---

## 📊 Dataset & Benchmark Evaluation

Evaluated on the **SisFall Benchmark Dataset** structured into fixed-length temporal windows ($256 \text{ time steps} \times 6 \text{ channels}$ at 50 Hz):

| Split | Windows | ADL (Class 0) | Pre-Fall (Class 1) | Fall (Class 2) |
| :--- | :---: | :---: | :---: | :---: |
| **Train Split** | 77,871 | 74,912 | 662 | 2,297 |
| **Validation Split** | 5,562 | 5,351 | 47 | 164 |
| **Test Split** | 9,270 | 8,918 | 79 | 273 |
| **Total** | **92,703** | **89,181 (96.2%)** | **788 (0.85%)** | **2,734 (2.95%)** |

### Benchmark Model Comparison

| Model Architecture | Parameters | Flash Footprint | ESP32 Latency | Fall Sensitivity | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Forest (Handcrafted) | 148,200 | 184 KB | 4.1 ms | 91.2% | 89.4% |
| HistGradientBoosting | 84,000 | 112 KB | 3.8 ms | 88.6% | 87.1% |
| Standard 1D-CNN | 42,100 | 48.6 KB | 16.8 ms | 93.4% | 91.8% |
| CNN-LSTM Hybrid | 89,400 | 98.2 KB | 38.4 ms | 94.1% | 92.5% |
| **Proposed TinyFallNet (SE-ResCNN)** | **24,850** | **28.4 KB** | **12.4 ms** | **98.4%** | **96.8%** |

---

## 📂 Project Repository Structure

```text
├── .gitignore                                           # Repository ignore rules
├── README.md                                            # Comprehensive project documentation
├── dashboard.html                                       # 9-section master publication dashboard
├── live_testing.html                                    # Real-time interactive testing suite
├── run_tinyml_fall_detection.py                         # Master model training, evaluation & profiling
├── sisfall_fall_detection.py                            # SisFall dataset parser and preprocessor
├── Review1_Report_Merged_FINAL.docx                     # Phase 1 finalized academic report
├── rev2_FINAL.docx                                      # Phase 2 finalized academic report
├── Review3_Unified_Master_Report_FINAL.docx             # Phase 3 finalized unified master report
├── Bharath_Palanisamy_Mettukadai_R3_Individual_Report.docx # Hardware & sensor individual report
├── Kaleeshwar_R3_Individual_Report.docx                 # ML model & TinyML individual report
├── Hiba_Fathima_R3_Individual_Report.docx               # Embedded testing & UI individual report
└── outputs/                                             # Evaluation outputs, metrics & charts
    ├── confusion_matrices_all_models.png
    ├── roc_and_pr_curves_tinyfallnet.png
    ├── tinyml_pareto_frontier.png
    ├── threshold_calibration_plot.png
    ├── severity_and_physiological_triage.png
    ├── model_benchmark_comparison.csv
    ├── tinyml_hardware_profiling.csv
    ├── threshold_calibration_study.csv
    └── loss_function_ablation_study.csv
```

---

## 🛠️ Getting Started

### 1. Interactive Testing Suite (No Installation Required)
Simply double-click or open either of the following files in any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox):
* [`live_testing.html`](live_testing.html): Test real-time parameter variations, preset scenarios, and inference probabilities.
* [`dashboard.html`](dashboard.html): Review benchmark performance graphs, ROC/PR curves, and TinyML Pareto trade-offs.

### 2. Python Environment Setup
To reproduce the model training, evaluation, and hardware profiling:
```bash
# Clone the repository
git clone https://github.com/kaleeshwar06/TinyML-Elderly-Fall-Detection.git
cd TinyML-Elderly-Fall-Detection

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

### 3. Run Benchmark Pipeline
```bash
python run_tinyml_fall_detection.py
```

---

## 📜 Academic Integrity & Project Scope Notice
This repository reflects the verified, reproducible software and embedded prototyping milestones completed for the project:
* 6-axis IMU sensor interfacing and magnitude extraction on the ESP32 microcontroller.
* TinyFallNet SE-ResCNN deep learning modeling, class-weighted training, and INT8 profiling on the SisFall benchmark dataset.
* Interactive real-time edge-simulation inference and testing suite.
* *Note: Hardware PPG/optical vitals sensors, physical on-chip BLE transmission, and extended battery life trials are strictly excluded from the verified scope.*

---

## 📄 License
This project is released under the [MIT License](LICENSE).
