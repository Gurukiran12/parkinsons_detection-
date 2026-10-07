import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import zipfile

from features import extract_features_from_audio, get_default_features
from voice_agent import predict_voice_sample
from spiral_analyzer import predict_single_drawing, batch_predict_zip, get_or_load_spiral_model
from multimodal_engine import run_multimodal_assessment
from model_trainer import train_and_save_pipeline, prepare_data, train_classical_models, train_cnn_ensemble, ensemble_predict_proba

# Pre-load or ensure spiral model is ready
if not os.path.exists("spiral_model.keras") and os.path.exists("spiral_model.keras.zip"):
    shutil.copy("spiral_model.keras.zip", "spiral_model.keras")

get_or_load_spiral_model("spiral_model.keras")

# Custom Medical UI Theme CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

body, .gradio-container {
    background: radial-gradient(circle at 10% 20%, #0d1b2a 0%, #080e18 90%) !important;
    color: #e2e8f0 !important;
}

.medical-hero {
    background: linear-gradient(135deg, rgba(20, 35, 60, 0.85) 0%, rgba(10, 20, 35, 0.95) 100%);
    border: 1px solid rgba(0, 210, 255, 0.2);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px -10px rgba(0, 210, 255, 0.15);
    text-align: center;
}

.medical-badge {
    background: rgba(0, 210, 255, 0.15);
    color: #00d2ff;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 12px;
    letter-spacing: 0.5px;
    border: 1px solid rgba(0, 210, 255, 0.3);
}

.gr-button-primary {
    background: linear-gradient(135deg, #0072ff 0%, #00d2ff 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3) !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 210, 255, 0.5) !important;
}

.gr-tab {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    border-radius: 10px 10px 0 0 !important;
}

.gr-tab.selected {
    color: #00d2ff !important;
    border-bottom: 3px solid #00d2ff !important;
    background: rgba(0, 210, 255, 0.05) !important;
}

.card-box {
    background: rgba(18, 30, 49, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
}

.caption-label, .thumbnail-caption, .gallery .caption, .gallery-item .caption {
    color: #ffffff !important;
    background: rgba(10, 20, 35, 0.95) !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
"""

def handle_voice_analysis(audio_input):
    res, fig, df, explanation = predict_voice_sample(audio_input)
    if res is None:
        return None, None, "⚠️ Please record or upload an audio clip."
    return fig, df, explanation


def handle_drawing_analysis(image_input):
    if image_input is None:
        return "⚠️ Please draw or upload a spiral/wave image.", None, None
    label, prob, conf_str, heatmap, fig_plot = predict_single_drawing(image_input)
    severity_badge = "🔴 HIGH SEVERITY (Severe Motor Tremor)" if prob >= 0.75 else ("🟡 MODERATE SEVERITY (Subtle Tremor)" if prob >= 0.50 else "🟢 HEALTHY CONTROL (Normal Coordination)")
    status_md = f"""### 🧠 Motor Tremor Assessment Report
• **Diagnostic Classification**: **{label}**
• **Parkinson's Probability**: **{prob*100:.1f}%**
• **Confidence / Risk Tier**: `{conf_str}`
• **Clinical Severity Index**: **{severity_badge}**
"""
    return status_md, heatmap, fig_plot


def handle_batch_zip(zip_file):
    return batch_predict_zip(zip_file)


def handle_multimodal(audio_in, drawing_in, weight):
    return run_multimodal_assessment(audio_in, drawing_in, weight)


def handle_dataset_training(csv_file):
    if csv_file is None:
        csv_path = "parkinsons.csv"
    else:
        csv_path = csv_file.name if hasattr(csv_file, "name") else csv_file

    try:
        results, _ = train_and_save_pipeline(csv_path)
        
        report = ["### 📊 Voice Model Benchmark Results:\n"]
        for m, acc in results.items():
            report.append(f"- **{m}**: `{acc*100:.2f}%` Accuracy")
        
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#131e2b')
        ax.set_facecolor('#0d1520')
        models_list = list(results.keys())
        accs = [results[m] for m in models_list]
        
        colors = ['#00d2ff' if 'CNN' in m else '#3b82f6' for m in models_list]
        bars = ax.barh(models_list, accs, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Accuracy Score", color='#a0aec0')
        ax.set_title("Machine Learning & 1D-CNN Model Comparison", color='white', fontweight='bold')
        ax.tick_params(colors='#a0aec0')
        ax.grid(True, linestyle='--', alpha=0.2, color='white')
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%",
                    ha='left', va='center', color='white', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        return "\n".join(report), fig
    except Exception as e:
        return f"Error training models: {e}", None


# Build Gradio Blocks Application
with gr.Blocks(title="Parkinson's Multimodal AI Diagnostic Suite") as demo:
    
    # Hero Banner
    gr.HTML("""
    <div class='medical-hero'>
        <div class='medical-badge'>🔬 CLINICAL DECISION SUPPORT SYSTEM</div>
        <h1 style='font-size: 2.5rem; margin: 0; color: #ffffff; font-weight: 800;'>
            Parkinson's Disease Multi-Modal AI Detection
        </h1>
        <p style='font-size: 1.15rem; color: #94a3b8; max-width: 800px; margin: 12px auto 0 auto; line-height: 1.6;'>
            Early, non-invasive automated detection platform combining <b>Real-Time Voice Acoustic Biomarkers</b> 
            and <b>2D-CNN Spiral & Wave Drawing Motor Tremor Analysis</b> into a unified diagnostic pipeline.
        </p>
    </div>
    """)

    with gr.Tabs():
        
        # TAB 1: Real-Time Voice Agent & Vocal Biomarkers
        with gr.Tab("🎙️ Real-Time Voice Agent"):
            gr.Markdown("""
            ### 🎙️ Phonation & Continuous Speech Acoustic Analysis
            *Instructions for Patient:* Click **Record from microphone** and speak a natural sentence (e.g., *"Today is a good day for a pleasant walk"*) **OR** sustain a clear vowel sound (e.g. **'aaaaah'**) for **3 to 5 seconds**.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    voice_mic_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Patient Voice Recording / Microphone")
                    voice_btn = gr.Button("🔍 Analyze Vocal Biomarkers", variant="primary")
                    
                    gr.Markdown("""
                    **Acoustic Features Evaluated:**
                    - Fundamental Frequency ($F_0, F_{hi}, F_{lo}$)
                    - Frequency Instability (Jitter: Local, RAP, PPQ, DDP)
                    - Amplitude Perturbation (Shimmer: Local, APQ3, APQ5, DDA)
                    - Noise-to-Harmonic (NHR) & Harmonics-to-Noise (HNR)
                    - Nonlinear Dynamics (RPDE, DFA, PPE)
                    """)
                    
                with gr.Column(scale=2):
                    voice_report_md = gr.Markdown("### 📊 Diagnostic Results will appear here...")
                    voice_plot_output = gr.Plot(label="Acoustic Signal & Biomarker Analysis")
                    voice_features_table = gr.DataFrame(label="Extracted 22 Clinical Voice Features", wrap=True)
                    
            voice_btn.click(
                handle_voice_analysis,
                inputs=[voice_mic_input],
                outputs=[voice_plot_output, voice_features_table, voice_report_md]
            )

        # TAB 2: Motor Tremor Assessment (Spiral / Wave Drawing)
        with gr.Tab("✍️ Motor Drawing Analysis"):
            gr.Markdown("""
            ### ✍️ Spiral & Wave Motor Tremor Assessment (2D CNN)
            Parkinsonian resting and kinetic tremors manifest as subtle irregularities, tremors, and distortions in hand-drawn spiral and wave patterns.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    drawing_input = gr.Image(sources=["upload", "clipboard"], type="pil", label="Upload Spiral or Wave Drawing Image")
                    drawing_btn = gr.Button("🧠 Analyze Drawing Pattern", variant="primary")
                    drawing_heatmap_out = gr.Image(label="Tremor Instability Heatmap Overlay")
                    
                with gr.Column(scale=2):
                    drawing_result_md = gr.Markdown("### 🔍 Drawing Classification will appear here...")
                    drawing_plot_out = gr.Plot(label="Kinematic Tremor Trace & Severity Gauge Meter")

            drawing_btn.click(
                handle_drawing_analysis,
                inputs=[drawing_input],
                outputs=[drawing_result_md, drawing_heatmap_out, drawing_plot_out]
            )

            gr.Markdown("---")
            gr.Markdown("### 🗂️ Batch Test Suite (`Spiral_Test.zip`)")
            with gr.Row():
                with gr.Column(scale=1):
                    zip_input = gr.File(label="Upload Spiral/Wave ZIP Test Dataset (.zip)", file_types=[".zip"], value="Spiral_Test.zip" if os.path.exists("Spiral_Test.zip") else None)
                    batch_btn = gr.Button("🚀 Run Batch Test on All Images")
                    batch_status_md = gr.Markdown("")
                with gr.Column(scale=2):
                    gallery_output = gr.Gallery(label="Batch Classification Gallery", columns=4, height=350)
            
            batch_btn.click(
                handle_batch_zip,
                inputs=[zip_input],
                outputs=[gallery_output, batch_status_md]
            )

        # TAB 3: Multimodal Joint Decision Support
        with gr.Tab("🧠 Multimodal Decision Engine"):
            gr.Markdown("""
            ### 🧠 Multi-Headed Joint Assessment (Vocal + Motor Fusion)
            Evaluates both physiological dimensions simultaneously for enhanced diagnostic sensitivity.
            """)
            
            with gr.Row():
                with gr.Column():
                    multi_audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="1. Patient Voice Recording")
                    multi_drawing = gr.Image(sources=["upload"], type="pil", label="2. Spiral / Wave Drawing")
                    voice_weight_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Voice vs Motor Weight Balance (Default: 50/50)")
                    multi_btn = gr.Button("⚡ Run Joint Multimodal Assessment", variant="primary")
                    
                with gr.Column():
                    multi_report_html = gr.HTML("<div style='padding:20px; text-align:center; color:#94a3b8;'>Upload modalities and click 'Run Joint Multimodal Assessment'</div>")
                    with gr.Row():
                        multi_voice_fig = gr.Plot(label="Acoustic Signal")
                        multi_draw_img = gr.Image(label="Tremor Heatmap")
                    multi_df = gr.DataFrame(label="Multimodal Diagnostic Decision Matrix")

            multi_btn.click(
                handle_multimodal,
                inputs=[multi_audio, multi_drawing, voice_weight_slider],
                outputs=[multi_report_html, multi_voice_fig, multi_draw_img, multi_df]
            )

        # TAB 4: Voice Dataset Training & Benchmark Lab
        with gr.Tab("📊 Voice ML Training Lab"):
            gr.Markdown("""
            ### 📊 Train & Benchmark Voice Models
            Trains classical classifiers (Random Forest, XGBoost, SVM, KNN) alongside a **1D-CNN Bagging Ensemble** with correlation clustering and SMOTE balancing.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    csv_upload = gr.File(label="Upload Parkinson's Dataset (.csv)", file_types=[".csv"], value="parkinsons.csv" if os.path.exists("parkinsons.csv") else None)
                    train_btn = gr.Button("⚡ Train All Models & Benchmark", variant="primary")
                    train_report_md = gr.Markdown("")
                with gr.Column(scale=2):
                    train_plot_out = gr.Plot(label="Model Accuracy Benchmark")

            train_btn.click(
                handle_dataset_training,
                inputs=[csv_upload],
                outputs=[train_report_md, train_plot_out]
            )

        # TAB 5: Clinical Guide & Methodology
        with gr.Tab("ℹ️ Clinical Reference & Abstract"):
            gr.Markdown("""
            ### 📖 Project Methodology & Clinical Context
            
            **Abstract Summary:**
            > Parkinson’s Disease (PD) is a chronic neurological disorder affecting motor functions, speech, and coordination. 
            > This automated system performs early detection using both **voice features** and **spiral/wave drawing patterns**.
            > - **Vocal Analysis**: Extracts acoustic parameters (Jitter, Shimmer, Pitch variations, HNR) reflecting vocal fold instability caused by hypokinetic dysarthria.
            > - **Motor Drawing Analysis**: Evaluates hand movement tremors and distortions in spiral/wave sketches using a 2D Convolutional Neural Network (CNN).
            > - **Multi-Headed Decision Fusion**: Combines acoustic and kinematic representations to provide an automated, non-invasive decision-support tool.

            ---
            
            ### 🔬 Acoustic Biomarker Reference Guide:
            | Biomarker | Healthy Threshold | Parkinsonian Indicator | Clinical Interpretation |
            | :--- | :--- | :--- | :--- |
            | **MDVP:Jitter(%)** | < 0.6% | > 1.0% | Micro-instability in vocal cord vibration frequency |
            | **MDVP:Shimmer** | < 3.0% | > 4.5% | Perturbation and instability in vocal amplitude/loudness |
            | **HNR (Harmonics-to-Noise)** | > 20 dB | < 18 dB | Breathiness, turbulence, and dysphonia in voice |
            | **PPE (Pitch Period Entropy)** | < 0.15 | > 0.25 | Impaired pitch regulation and monotonic speech |
            """)

    gr.HTML("<footer style='text-align:center; padding: 20px; color:#64748b; font-size:0.85rem;'>Parkinson's Disease Multimodal Detection Platform • Built with Python, TensorFlow & Gradio</footer>")
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
        css=custom_css,
        inbrowser=True
    )



