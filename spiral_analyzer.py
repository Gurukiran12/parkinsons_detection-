import os
import cv2
import zipfile
import shutil
import numpy as np
from PIL import Image
try:
    import keras
    from keras.applications.mobilenet_v2 import preprocess_input
    load_keras_model = keras.models.load_model
except ImportError:
    try:
        import tensorflow as tf
        preprocess_input = getattr(tf.keras.applications, "mobilenet_v2").preprocess_input
        load_keras_model = getattr(tf.keras.models, "load_model")
    except Exception:
        preprocess_input = lambda x: (x / 127.5) - 1.0
        load_keras_model = None

_LOADED_CNN = None
_INPUT_SIZE = (224, 224)


def get_or_load_spiral_model(model_path="spiral_model.keras"):
    global _LOADED_CNN, _INPUT_SIZE
    if _LOADED_CNN is not None:
        return _LOADED_CNN, _INPUT_SIZE

    if not os.path.exists(model_path):
        if os.path.exists("spiral_model.keras.zip"):
            import shutil
            shutil.copy("spiral_model.keras.zip", "spiral_model.keras")

    if os.path.exists(model_path):
        try:
            _LOADED_CNN = load_keras_model(model_path, compile=False)
            shape = _LOADED_CNN.input_shape
            if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
                _INPUT_SIZE = (shape[1], shape[2])
            return _LOADED_CNN, _INPUT_SIZE
        except Exception as e:
            print(f"Failed loading {model_path}: {e}")
    return None, _INPUT_SIZE
import matplotlib.pyplot as plt


def create_drawing_clinical_plot(rgb_img, prob, label, severity_level):
    """
    Creates a multi-panel visual report for motor drawing analysis:
    1. Preprocessed Input Drawing
    2. Kinematic Tremor Edge Contour Map
    3. Severity Index Meter / Gauge
    """
    fig = plt.figure(figsize=(12, 4), facecolor='#131e2b')

    # Panel 1: Original Drawing
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor('#0d1520')
    ax1.imshow(rgb_img)
    ax1.set_title("Input Drawing Sketch", color='white', fontsize=11, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Laplacian Kinematic Tremor Map
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor('#0d1520')
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    laplacian = np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))
    ax2.imshow(laplacian, cmap='inferno')
    ax2.set_title("Kinematic Micro-Tremor Trace", color='white', fontsize=11, fontweight='bold')
    ax2.axis('off')

    # Panel 3: Severity Gauge Bar
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor('#0d1520')
    
    stages = ["Normal", "Mild", "Moderate", "Severe"]
    thresholds = [0.25, 0.50, 0.75, 1.0]
    colors = ["#00e676", "#ffd600", "#ff9100", "#ff1744"]
    
    # Draw severity gradient bar
    ax3.barh([0], [0.35], color="#00e676", height=0.4, label="Healthy / Normal (<35%)")
    ax3.barh([0], [0.25], left=[0.35], color="#ffd600", height=0.4, label="Mild Tremor (35-60%)")
    ax3.barh([0], [0.20], left=[0.60], color="#ff9100", height=0.4, label="Moderate Tremor (60-80%)")
    ax3.barh([0], [0.20], left=[0.80], color="#ff1744", height=0.4, label="Severe Tremor (>80%)")
    
    # Mark current prediction needle
    ax3.axvline(prob, color='white', linewidth=3.0, linestyle='-')
    ax3.plot([prob], [0], marker='v', markersize=14, color='white')
    
    ax3.set_xlim(0, 1.05)
    ax3.set_ylim(-0.5, 0.8)
    ax3.set_yticks([])
    ax3.set_xlabel("Disease Severity Index (%)", color='#a0aec0', fontsize=10, fontweight='bold')
    ax3.set_title(f"Severity Score: {prob*100:.1f}% ({severity_level})", color='white', fontsize=11, fontweight='bold')
    ax3.tick_params(colors='#a0aec0')
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7, facecolor='#0d1520', edgecolor='#2d3748', labelcolor='white')

    plt.tight_layout()
    return fig


def predict_single_drawing(image_input, model_path="spiral_model.keras"):
    """
    Predicts Parkinson's risk from a single drawing image (PIL Image, numpy array, or file path).
    Returns: label, probability, confidence_str, heatmap_img, fig_plot
    """
    model, target_size = get_or_load_spiral_model(model_path)
    if model is None:
        return "Model not found", 0.5, "0.00%", None, None

    try:
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, dict):
            if "composite" in image_input and image_input["composite"] is not None:
                img = Image.fromarray(image_input["composite"]).convert("RGB")
            elif "background" in image_input and image_input["background"] is not None:
                img = Image.fromarray(image_input["background"]).convert("RGB")
            else:
                img = Image.fromarray(list(image_input.values())[0]).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        # Resize for CNN
        img_resized = img.resize(target_size)
        arr = np.array(img_resized).astype(np.float32)
        arr_preprocessed = preprocess_input(arr)
        arr_batch = np.expand_dims(arr_preprocessed, 0)

        pred = model.predict(arr_batch, verbose=0)

        if pred.shape[-1] == 1:
            prob = float(pred[0][0])
            if prob >= 0.5:
                label = "Parkinson's Indicators Detected"
                confidence = prob
                if prob >= 0.80:
                    severity = "Severe Tremor Pattern"
                    risk = "High Risk"
                elif prob >= 0.60:
                    severity = "Moderate Tremor Pattern"
                    risk = "Moderate Risk"
                else:
                    severity = "Mild / Early Tremor Pattern"
                    risk = "Mild Risk"
            else:
                label = "Healthy / Normal Drawing"
                confidence = 1.0 - prob
                severity = "Normal Motor Coordination"
                risk = "Low Risk"
        else:
            idx = int(np.argmax(pred))
            confidence = float(np.max(pred))
            prob = float(pred[0][1]) if pred.shape[-1] > 1 else confidence
            if idx == 1:
                label = "Parkinson's Indicators Detected"
                severity = "Severe Tremor Pattern" if prob > 0.75 else "Moderate Tremor Pattern"
                risk = "High Risk"
            else:
                label = "Healthy / Normal Drawing"
                severity = "Normal Motor Coordination"
                risk = "Low Risk"

        rgb_for_plot = np.array(img.resize((300, 300)))
        heatmap_img = generate_tremor_heatmap(rgb_for_plot)
        fig_plot = create_drawing_clinical_plot(rgb_for_plot, prob, label, severity)

        return label, prob, f"{confidence*100:.2f}% ({risk})", heatmap_img, fig_plot

    except Exception as e:
        print(f"Drawing prediction error: {e}")
        return f"Error: {e}", 0.5, "N/A", None, None


def generate_tremor_heatmap(rgb_img):
    """
    Applies Laplacian / Sobel tremor frequency detection overlay to visualize hand drawing tremor.
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.uint8(np.absolute(laplacian))
    heatmap = cv2.applyColorMap(laplacian_abs * 2, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb_img, 0.65, heatmap, 0.35, 0)
    return overlay


def batch_predict_zip(zip_file, model_path="spiral_model.keras"):
    """
    Extracts ZIP archive and predicts all contained images.
    Returns: gallery list [(image_path, caption)], summary_text
    """
    model, target_size = get_or_load_spiral_model(model_path)
    if model is None:
        return [], "Please load or supply spiral_model.keras first."

    if zip_file is None:
        return [], "No ZIP file uploaded."

    extract_dir = "prediction_images"
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)

    zip_path = zip_file if isinstance(zip_file, str) else zip_file.name
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_paths = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(root, f))
    image_paths = sorted(image_paths)

    if not image_paths:
        return [], "No images found in uploaded ZIP."

    gallery = []
    healthy = 0
    parkinson = 0

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img_res = img.resize(target_size)
            arr = np.array(img_res).astype(np.float32)
            arr = preprocess_input(arr)
            arr = np.expand_dims(arr, 0)

            pred = model.predict(arr, verbose=0)
            if pred.shape[-1] == 1:
                prob = float(pred[0][0])
                if prob >= 0.5:
                    label = "Parkinson's"
                    confidence = prob
                    parkinson += 1
                else:
                    label = "Healthy"
                    confidence = 1 - prob
                    healthy += 1
            else:
                idx = np.argmax(pred)
                confidence = float(np.max(pred))
                if idx == 1:
                    label = "Parkinson's"
                    parkinson += 1
                else:
                    label = "Healthy"
                    healthy += 1

            filename = os.path.basename(path)
            gallery.append((path, f"{filename}\n{label} ({confidence*100:.1f}%)"))
        except Exception as e:
            print(f"Error on {path}: {e}")

    total = len(gallery)
    summary = f"""✅ **Batch Analysis Complete**
• **Total Drawings Processed**: {total}
• **Healthy / Normal**: {healthy} ({healthy/total*100:.1f}%)
• **Parkinson's Indicators**: {parkinson} ({parkinson/total*100:.1f}%)
"""
    return gallery, summary
