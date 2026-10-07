# 🔬 Parkinson's Disease Multi-Modal AI Detection Suite

A multi-modal AI clinical decision support platform for non-invasive, early detection of Parkinson's Disease combining **Real-Time Voice Acoustic Biomarkers** and **2D-CNN Spiral & Wave Drawing Motor Tremor Analysis** into a unified diagnostic pipeline.

---

## 🌟 Key Features

1. **🎙️ Real-Time Voice Biomarker Assessment**:
   - Computes 22 clinical acoustic features via Praat Parselmouth & Librosa.
   - Extracts Fundamental Frequency ($F_0$), Vocal Jitter (Frequency Instability), Vocal Shimmer (Amplitude Perturbation), Harmonics-to-Noise Ratio (HNR), and Pitch Period Entropy (PPE).
   - Powered by a 1D-CNN Bagging Ensemble and classical machine learning models (XGBoost, Random Forest, SVM).

2. **🌀 Spiral & Wave Drawing Motor Analysis**:
   - Inspects motor tremors and kinematic drawing irregularities using a fine-tuned MobileNetV2 2D-CNN.
   - Computes drawing contour geometric smoothness, entropy, and spatial tremor patterns.

3. **🩺 Multimodal Fusion Decision Engine**:
   - Integrates voice phonation analysis, motor drawing kinematics, and clinical questionnaire data with weighted probabilistic fusion.

4. **⚙️ In-App ML Model Training Lab**:
   - Train and benchmark classical ML classifiers and Deep Learning models on `parkinsons.csv` with real-time ROC curves and performance metrics.

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/gurukiranintech-png/parkinsons_detection.git
cd parkinsons_detection
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:7860`.

---

## 📁 Repository Structure

```
├── app.py                     # Main Gradio Multimodal Web Interface
├── features.py                # Acoustic feature extraction (Praat & Librosa)
├── voice_agent.py             # Voice phonation assessment & clinical plotting
├── spiral_analyzer.py         # MobileNetV2 CNN & drawing kinematic analysis
├── multimodal_engine.py       # Multimodal probabilistic fusion engine
├── model_trainer.py           # Training pipeline for ML & 1D-CNN ensemble
├── parkinsons.csv             # Clinical vocal dysphonia dataset
├── spiral_model.keras         # Pretrained MobileNetV2 drawing model
├── trained_models/            # Serialized voice ensemble models & scalers
└── requirements.txt           # Project dependencies
```

---

## 🛠️ Tech Stack
- **Frameworks**: Python, TensorFlow / Keras, Scikit-Learn, XGBoost, Gradio
- **Audio Processing**: Praat-Parselmouth, Librosa, SciPy, SoundFile
- **Computer Vision**: OpenCV, Pillow (PIL), Matplotlib
