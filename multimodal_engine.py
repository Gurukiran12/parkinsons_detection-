import numpy as np
import pandas as pd
from voice_agent import predict_voice_sample
from spiral_analyzer import predict_single_drawing


def run_multimodal_assessment(audio_input, drawing_input, voice_weight=0.5):
    """
    Jointly evaluates Patient Vocal Biometrics & Spiral/Wave Motor Drawings.
    Returns:
    - joint_summary_html: Structured Clinical Report with Risk Stratification
    - voice_fig: Acoustic plot
    - drawing_heatmap: Motor tremor drawing heatmap
    - metrics_df: Summary comparison metrics
    """
    has_voice = audio_input is not None
    has_drawing = drawing_input is not None

    if not has_voice and not has_drawing:
        return (
            "<div style='padding:20px; text-align:center; color:#ffb74d;'>⚠️ Please provide at least one diagnostic modality (Voice Recording or Drawing).</div>",
            None,
            None,
            None
        )

    voice_prob = None
    voice_res = None
    voice_fig = None
    voice_df = None
    drawing_prob = None
    drawing_label = None
    drawing_heatmap = None

    # Process Voice Modality
    if has_voice:
        try:
            voice_res, voice_fig, voice_df, _ = predict_voice_sample(audio_input)
            if voice_res is not None:
                voice_prob = voice_res["probability"]
        except Exception as e:
            print(f"Voice multimodal error: {e}")

    # Process Drawing Modality
    if has_drawing:
        try:
            drawing_label, drawing_prob, _, drawing_heatmap = predict_single_drawing(drawing_input)
        except Exception as e:
            print(f"Drawing multimodal error: {e}")

    # Compute Joint Multimodal Probability
    if voice_prob is not None and drawing_prob is not None:
        w_v = float(voice_weight)
        w_d = 1.0 - w_v
        joint_prob = (w_v * voice_prob) + (w_d * drawing_prob)
        modality_status = "Combined Multimodal Fusion (Voice + Motor Drawing)"
    elif voice_prob is not None:
        joint_prob = voice_prob
        modality_status = "Unimodal Assessment (Acoustic Voice Only)"
    else:
        joint_prob = drawing_prob
        modality_status = "Unimodal Assessment (Motor Drawing Only)"

    # Risk Stratification
    if joint_prob >= 0.65:
        risk_tier = "HIGH RISK"
        badge_color = "#ff4444"
        bg_badge = "rgba(255, 68, 68, 0.2)"
        clinical_advice = "Notable vocal instability and hand-motor tremors detected. Clinical neurological consultation (MDS-UPDRS assessment) is strongly advised."
    elif joint_prob >= 0.40:
        risk_tier = "MODERATE / BORDERLINE RISK"
        badge_color = "#ffbb33"
        bg_badge = "rgba(255, 187, 51, 0.2)"
        clinical_advice = "Mild micro-tremor or acoustic variations detected. Periodic follow-up and repeat assessment recommended."
    else:
        risk_tier = "LOW RISK / NORMAL"
        badge_color = "#00C851"
        bg_badge = "rgba(0, 200, 81, 0.2)"
        clinical_advice = "Acoustic parameters and spiral drawing patterns lie within healthy normative thresholds."

    # Build Comprehensive Medical HTML Card
    v_val_str = f"{voice_prob*100:.1f}%" if voice_prob is not None else "Not Provided"
    d_val_str = f"{drawing_prob*100:.1f}%" if drawing_prob is not None else "Not Provided"

    html_report = f"""
    <div style="background: linear-gradient(135deg, #101c2c, #162a40); border: 1px solid rgba(255,255,255,0.15); border-radius: 16px; padding: 24px; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 16px; margin-bottom: 20px;">
            <div>
                <h2 style="margin: 0; font-size: 1.6rem; color: #64b5f6; font-weight: 700;">🏥 Comprehensive Multimodal Diagnostic Summary</h2>
                <span style="font-size: 0.85rem; color: #90caf9;">Mode: {modality_status}</span>
            </div>
            <div style="background: {bg_badge}; border: 1.5px solid {badge_color}; color: {badge_color}; padding: 8px 18px; border-radius: 30px; font-weight: 800; font-size: 1.05rem; letter-spacing: 0.5px;">
                {risk_tier}
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px;">
            <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 12px; border-left: 4px solid #00d2ff;">
                <div style="font-size: 0.85rem; color: #a0aec0; text-transform: uppercase;">Joint PD Probability</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-top: 4px;">{joint_prob*100:.1f}%</div>
                <div style="font-size: 0.75rem; color: #90caf9;">Combined Multi-Head Score</div>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 12px; border-left: 4px solid #ab47bc;">
                <div style="font-size: 0.85rem; color: #a0aec0; text-transform: uppercase;">Vocal Biomarker Index</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-top: 4px;">{v_val_str}</div>
                <div style="font-size: 0.75rem; color: #ce93d8;">Acoustic Jitter/Shimmer/HNR</div>
            </div>

            <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 12px; border-left: 4px solid #ffca28;">
                <div style="font-size: 0.85rem; color: #a0aec0; text-transform: uppercase;">Motor Tremor Index</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-top: 4px;">{d_val_str}</div>
                <div style="font-size: 0.75rem; color: #ffe082;">2D CNN Spiral / Wave Analysis</div>
            </div>
        </div>

        <div style="background: rgba(0,0,0,0.25); border-radius: 12px; padding: 16px; margin-top: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #90caf9; font-size: 1rem;">🩺 Clinical Decision Support Recommendation:</h4>
            <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: #e0e0e0;">
                {clinical_advice}
            </p>
        </div>
    </div>
    """

    summary_table = {
        "Modality": ["Voice Acoustic Stream", "Spiral/Wave Motor Pattern", "Multimodal Fusion Decision"],
        "Sub-Score / Probability": [v_val_str, d_val_str, f"{joint_prob*100:.1f}%"],
        "Risk Stratification": [
            "Normal" if (voice_prob is not None and voice_prob < 0.5) else ("Abnormal" if voice_prob is not None else "N/A"),
            "Normal" if (drawing_prob is not None and drawing_prob < 0.5) else ("Abnormal" if drawing_prob is not None else "N/A"),
            risk_tier
        ]
    }
    df_metrics = pd.DataFrame(summary_table)

    return html_report, voice_fig, drawing_heatmap, df_metrics
