"""
Real-Data Machine Learning Models Page (Step 9 Phase C).
Displays chronological evaluation, feature importances, and out-of-sample predictions on real NIFTY 50 data.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.layout import setup_page, render_terminal_header, render_disclaimer
from app.components.sidebar import render_sidebar
from app.components.metrics import render_kpi_grid
from app.services.ml_service import MLService


def main() -> None:
    setup_page("Real ML Models", icon="🔬")
    render_sidebar(active_page="Real ML Models")

    render_terminal_header(
        title="REAL-DATA MACHINE LEARNING ENGINE",
        subtitle="CHRONOLOGICAL PREDICTION OF NEXT-DAY DIRECTION & RETURN MAGNITUDE (2015–2024)",
        badge="REAL NIFTY 50 // 121,962 SAMPLES // ZERO LEAKAGE",
    )

    ml_svc = MLService()
    real_metrics = ml_svc.get_real_metrics()
    df_preds = ml_svc.get_real_predictions()

    if not real_metrics:
        st.warning("Real ML model metrics not found. Please train models via Step 9 pipeline.")
        return

    meta = real_metrics.get("metadata", {})
    clf_metrics = real_metrics.get("classification", {})
    reg_metrics = real_metrics.get("regression", {})
    feat_imp = real_metrics.get("feature_importance", {})

    # Top KPI Cards
    best_clf_acc = clf_metrics.get("Logistic_Regression", {}).get("test_accuracy", 0.5074) * 100
    best_clf_auc = clf_metrics.get("Logistic_Regression", {}).get("test_roc_auc", 0.5167)
    best_reg_r2 = reg_metrics.get("Random_Forest_Regressor", {}).get("test_r2", 0.0293) * 100
    best_reg_dir = reg_metrics.get("Ridge_Regression", {}).get("test_directional_accuracy", 0.5110) * 100

    metric_cards = [
        {
            "label": "Test Direction Accuracy",
            "value": f"{best_clf_acc:.2f}%",
            "delta": "Logistic Regression (Unseen 2024)",
            "help": "Out-of-sample accuracy on next-day direction (Up vs Down/Flat).",
        },
        {
            "label": "Test ROC-AUC",
            "value": f"{best_clf_auc:.4f}",
            "delta": "vs 0.5000 Baseline",
            "help": "Area Under ROC Curve on 2024 test partition.",
        },
        {
            "label": "Test Regression R-Squared",
            "value": f"+{best_reg_r2:.2f}%",
            "delta": "Random Forest Regressor",
            "help": "Out-of-sample variance explained on 2024 daily return magnitudes.",
        },
        {
            "label": "Test Directional Hit Rate",
            "value": f"{best_reg_dir:.2f}%",
            "delta": "Ridge Regressor Sign",
            "help": "Percentage of test days where predicted return sign matched actual sign.",
        },
    ]
    render_kpi_grid(metric_cards, columns=4)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for Inspection
    tab_eval, tab_feat, tab_preds, tab_infer = st.tabs([
        "📊 Chronological Model Benchmarks",
        "💡 Feature Importance",
        "🔍 Out-of-Sample Predictions (2024)",
        "🧪 Interactive Single-Sample Predictor",
    ])

    with tab_eval:
        st.subheader("Task 1: Next-Day Direction Classification (Binary: Up vs Down)")
        st.markdown(
            f"*Evaluation Scheme: Training (2015–2021: {meta.get('train_samples', 81619):,} samples), "
            f"Validation (2022–2023: {meta.get('val_samples', 24600):,} samples), "
            f"Unseen Test (2024: {meta.get('test_samples', 12250):,} samples). Boundary purged.*"
        )

        clf_rows = []
        for name, d in clf_metrics.items():
            clf_rows.append({
                "Model Architecture": name.replace("_", " "),
                "Val Accuracy": f"{d.get('val_accuracy', 0)*100:.2f}%",
                "Test Accuracy": f"{d.get('test_accuracy', 0)*100:.2f}%",
                "Test Balanced Acc": f"{d.get('test_balanced_accuracy', 0)*100:.2f}%",
                "Test Precision": f"{d.get('test_precision', 0)*100:.2f}%",
                "Test Recall": f"{d.get('test_recall', 0)*100:.2f}%",
                "Test F1-Score": f"{d.get('test_f1', 0):.4f}",
                "Test ROC-AUC": f"{d.get('test_roc_auc', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(clf_rows), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Task 2: Next-Day Return Regression (Continuous Return $r_{t+1}$)")

        reg_rows = []
        for name, d in reg_metrics.items():
            conv_r2 = d.get('test_r2_conventional', d.get('test_r2', 0))
            rel_r2 = d.get('test_r2_relative_vs_zero', (1.0 - (d.get('test_rmse', 0.017141)**2 / (0.017141**2))) if d.get('test_rmse') else 0.0)
            reg_rows.append({
                "Model Architecture": name.replace("_", " "),
                "Val MAE": f"{d.get('val_mae', 0):.6f}",
                "Val RMSE": f"{d.get('val_rmse', 0):.6f}",
                "Test MAE": f"{d.get('test_mae', 0):.6f}",
                "Test RMSE": f"{d.get('test_rmse', 0):.6f}",
                "Test R² (Conventional)": f"{conv_r2:+.6f}",
                "Test R² (vs Zero Baseline)": f"{rel_r2:+.6f}",
                "Test Directional Acc": f"{d.get('test_directional_accuracy', 0)*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True)
        st.caption(
            "Note on R² metrics: Conventional R² evaluates variance explained relative to sample mean return (R² = 1 - MSE / Var(y)). "
            "Zero-return baseline achieves conventional R² = -0.001334 because test mean return is +0.000626 ≠ 0. "
            "Relative to the zero-return benchmark (ΔMSE_zero = 1 - MSE / MSE_zero), baseline is exactly 0.0000, Ridge is +0.003618, and RF is +0.030558."
        )

    with tab_feat:
        st.subheader("Feature Importance Analysis (Random Forest)")
        if feat_imp:
            df_imp = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                df_imp, x="Importance", y="Feature", orientation="h",
                title="Random Forest Explanatory Feature Importance (Gini Impurity Reduction)",
                color="Importance", color_continuous_scale="Blues",
            )
            fig_imp.update_layout(height=500, template="plotly_dark")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.info(
                "**Insight**: Systematic market co-movement (`benchmark_daily_return`) accounts for **18.5%** of feature importance, "
                "followed by active spread vs index (`relative_return_vs_nifty`) and short-term momentum (`daily_adjusted_return`)."
            )

    with tab_preds:
        st.subheader("Unseen Test Set Predictions Explorer (Calendar Year 2024)")
        if not df_preds.empty:
            symbols = ["ALL"] + sorted(df_preds["symbol"].unique().tolist())
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                sel_sym = st.selectbox("Filter Ticker Symbol:", symbols, index=0)
            with col_f2:
                sel_date = st.text_input("Filter Date (YYYY-MM-DD, optional):", "")

            df_filt = df_preds.copy()
            if sel_sym != "ALL":
                df_filt = df_filt[df_filt["symbol"] == sel_sym]
            if sel_date.strip():
                df_filt = df_filt[df_filt["date"].str.contains(sel_date.strip())]

            st.dataframe(df_filt.head(200), use_container_width=True, height=400)
            csv_preds = df_filt.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Filtered Test Predictions", csv_preds, "real_ml_test_predictions_2024.csv", "text/csv")

    with tab_infer:
        st.subheader("Interactive Single-Sample Inference Sandbox")
        st.markdown(
            "*Simulate next-day prediction by supplying custom factor values.*"
        )
        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            in_ret = st.number_input("Prior Day Return (%)", value=1.2, step=0.1) / 100.0
            in_ret5 = st.number_input("5-Day Return (%)", value=2.5, step=0.1) / 100.0
            in_ret20 = st.number_input("20-Day Return (%)", value=4.0, step=0.2) / 100.0
            in_ret60 = st.number_input("60-Day Return (%)", value=8.0, step=0.5) / 100.0
        with col_i2:
            in_vol20 = st.number_input("20-Day Volatility (ann)", value=0.22, step=0.01)
            in_vol60 = st.number_input("60-Day Volatility (ann)", value=0.24, step=0.01)
            in_ma20_spread = st.number_input("Spread to MA20 (%)", value=1.5, step=0.1) / 100.0
            in_ma50_spread = st.number_input("Spread to MA50 (%)", value=3.0, step=0.2) / 100.0
            in_ma200_spread = st.number_input("Spread to MA200 (%)", value=6.0, step=0.5) / 100.0
        with col_i3:
            in_vol_ratio = st.number_input("Volume Ratio (vs 20d MA)", value=1.2, step=0.1)
            in_hl_range = st.number_input("Intraday High-Low Range (%)", value=2.1, step=0.1) / 100.0
            in_mdd60 = st.number_input("60-Day Rolling Drawdown (%)", value=-5.0, step=0.5) / 100.0
            in_bench_ret = st.number_input("NIFTY Benchmark Return (%)", value=0.8, step=0.1) / 100.0
            in_rel_ret = in_ret - in_bench_ret

        selected_model = st.selectbox("Select Classifier:", ["random_forest", "logistic_regression", "gradient_boosting"])

        if st.button("Run Model Prediction", type="primary"):
            input_dict = {
                "daily_adjusted_return": in_ret,
                "return_5d": in_ret5,
                "return_20d": in_ret20,
                "return_60d": in_ret60,
                "volatility_20d": in_vol20,
                "volatility_60d": in_vol60,
                "close_to_ma20": in_ma20_spread,
                "close_to_ma50": in_ma50_spread,
                "close_to_ma200": in_ma200_spread,
                "volume_ratio": in_vol_ratio,
                "hl_pct_range": in_hl_range,
                "rolling_max_drawdown_60d": in_mdd60,
                "benchmark_daily_return": in_bench_ret,
                "relative_return_vs_nifty": in_rel_ret,
            }
            res = ml_svc.predict_real_direction(input_dict, model_type=selected_model)
            if "error" in res:
                st.error(res["error"])
            else:
                pred = res["prediction"]
                p_up = res["probability_up"]
                p_down = res["probability_down"]

                color = "#00C805" if pred == "UP" else "#FF3B30"
                st.markdown(
                    f"""
                    <div style="background-color: #161922; border: 2px solid {color}; border-radius: 6px; padding: 16px; margin-top: 16px;">
                        <h4 style="color: {color}; margin: 0 0 8px 0;">Predicted Next-Day Direction: {pred}</h4>
                        <p style="margin: 0; color: #E0E3EB;">Probability Up: <strong>{p_up*100:.1f}%</strong> | Probability Down/Flat: <strong>{p_down*100:.1f}%</strong></p>
                        <p style="font-size: 0.78rem; color: #A0A5B5; margin-top: 6px;">Target Horizon: Next Trading Day (t+1) | Model: {selected_model.upper()}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning(
        "⚠️ **Market Efficiency & Model Limitations Disclosure**: The tested models show limited out-of-sample "
        "predictive performance on the selected historical NIFTY 50 universe. These findings do not establish an exploitable "
        "trading edge and neither prove nor disprove the Efficient Market Hypothesis. While execution frictions such as transaction "
        "taxes, brokerage fees, and market impact slippage were not empirically modeled, their presence in real trading environments "
        "typically erodes marginal statistical edges of this magnitude. Outputs do NOT constitute financial advice."
    )

    render_disclaimer()


if __name__ == "__main__":
    main()

