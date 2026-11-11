"""
Page 3: Machine Learning Analytics.
Presents chronological out-of-sample evaluation for Regression, Classification,
and K-Means Clustering models, feature importances, and interactive inference simulation.
"""

import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.layout import setup_page, render_terminal_header, render_disclaimer
from app.components.sidebar import render_sidebar
from app.components.metrics import render_kpi_grid
from app.components.charts import create_confusion_matrix_chart, create_cluster_scatter
from app.components.tables import render_downloadable_table
from app.services.ml_service import MLService


def main() -> None:
    setup_page("Machine Learning", icon="🤖")
    render_sidebar(active_page="Machine Learning")

    render_terminal_header(
        title="MACHINE LEARNING ANALYTICS & PREDICTION MODELS",
        subtitle="CHRONOLOGICAL TRAIN/TEST EVALUATION & K-MEANS PROFILES (PHASE 4)",
        badge="ML PIPELINE",
    )

    svc = MLService()
    metrics = svc.get_metrics()

    if not metrics:
        st.error("Model metrics file not found in data/processed/ml/model_metrics.json. Run scripts/train_models.py.")
        return

    # Regression and Classification Metrics
    reg_metrics = metrics.get("regression", {})
    clf_metrics = metrics.get("classification", {})
    cluster_metrics = metrics.get("clustering", {})

    rf_clf = clf_metrics.get("random_forest_classifier", {})
    lr_clf = clf_metrics.get("logistic_regression", {})
    base_clf = clf_metrics.get("majority_class_baseline", {})

    rf_reg = reg_metrics.get("random_forest_regressor", {})
    lr_reg = reg_metrics.get("linear_regression", {})

    # Top KPI Metrics Row
    kpis = [
        {"label": "Best Classifier Accuracy", "value": f"{rf_clf.get('accuracy', 0.0)*100:.2f}%", "delta": "Random Forest", "help": "Out-of-sample directional accuracy"},
        {"label": "Baseline Majority Accuracy", "value": f"{base_clf.get('accuracy', 0.0)*100:.2f}%", "delta": "Majority Class", "delta_color": "off"},
        {"label": "Classification Macro F1", "value": f"{rf_clf.get('f1_macro', 0.0):.4f}", "delta": "3-Class (Bull/Bear/Neut)"},
        {"label": "Linear Regressor MAE", "value": f"{lr_reg.get('mae', 0.0):.6f}", "delta": f"RMSE: {lr_reg.get('rmse', 0.0):.5f}"},
        {"label": "K-Means Chosen Clusters", "value": f"K = {cluster_metrics.get('chosen_k', 3)}", "delta": "Risk/Return Profiling"},
    ]
    render_kpi_grid(kpis, columns=5)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for ML Modules
    tab_clf, tab_reg, tab_cluster, tab_infer = st.tabs([
        "🎯 Market Direction Classification",
        "📉 Return Regression Analysis",
        "🧩 Stock Clustering (K-Means)",
        "🔮 Interactive Model Inference",
    ])

    # --- TAB 1: CLASSIFICATION ---
    with tab_clf:
        st.markdown("#### 🎯 Market Direction Classification Models (Chronological Test Set)")
        st.caption("Target classes: Bullish (Return > +0.5%), Bearish (Return < -0.5%), Neutral (Between -0.5% and +0.5%).")

        col_clf_tbl, col_clf_cm = st.columns([0.5, 0.5])

        with col_clf_tbl:
            st.markdown("##### 📊 Model Comparison Benchmark")
            comparison_data = [
                {
                    "Model": "Random Forest Classifier",
                    "Accuracy": f"{rf_clf.get('accuracy', 0.0)*100:.2f}%",
                    "Macro F1": f"{rf_clf.get('f1_macro', 0.0):.4f}",
                    "Weighted F1": f"{rf_clf.get('f1_weighted', 0.0):.4f}",
                    "Precision": f"{rf_clf.get('precision_macro', 0.0):.4f}",
                    "Recall": f"{rf_clf.get('recall_macro', 0.0):.4f}",
                },
                {
                    "Model": "Logistic Regression",
                    "Accuracy": f"{lr_clf.get('accuracy', 0.0)*100:.2f}%",
                    "Macro F1": f"{lr_clf.get('f1_macro', 0.0):.4f}",
                    "Weighted F1": f"{lr_clf.get('f1_weighted', 0.0):.4f}",
                    "Precision": f"{lr_clf.get('precision_macro', 0.0):.4f}",
                    "Recall": f"{lr_clf.get('recall_macro', 0.0):.4f}",
                },
                {
                    "Model": "Majority Class Baseline",
                    "Accuracy": f"{base_clf.get('accuracy', 0.0)*100:.2f}%",
                    "Macro F1": f"{base_clf.get('f1_macro', 0.0):.4f}",
                    "Weighted F1": f"{base_clf.get('f1_weighted', 0.0):.4f}",
                    "Precision": f"{base_clf.get('precision_macro', 0.0):.4f}",
                    "Recall": f"{base_clf.get('recall_macro', 0.0):.4f}",
                },
            ]
            st.dataframe(comparison_data, use_container_width=True, hide_index=True)

            # Feature Importance
            st.markdown("##### 🌟 Random Forest Feature Importance")
            feat_imp = svc.get_feature_importances()
            if not feat_imp.empty:
                fig_imp = px.bar(
                    feat_imp,
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                    height=280,
                )
                fig_imp.update_layout(
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#161922",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=40, r=40, t=20, b=30),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        with col_clf_cm:
            st.markdown("##### 🔲 Random Forest Confusion Matrix")
            cm = rf_clf.get("confusion_matrix", [])
            labels = rf_clf.get("labels", ["Bearish", "Neutral", "Bullish"])
            if cm:
                fig_cm = create_confusion_matrix_chart(cm, labels)
                st.plotly_chart(fig_cm, use_container_width=True)

        # Out-of-sample Test Predictions Table
        st.markdown("<br>", unsafe_allow_html=True)
        clf_preds = svc.get_classification_predictions()
        if not clf_preds.empty:
            render_downloadable_table(
                df=clf_preds,
                title="Test-Set Out-of-Sample Classification Predictions",
                filename="classification_test_predictions.csv",
                height=260,
            )

    # --- TAB 2: REGRESSION ---
    with tab_reg:
        st.markdown("#### 📉 Return Regression Models (Predicting Next-Day Return)")
        st.caption("Chronological train (80%) vs test (20%) split to avoid forward-looking lookahead bias.")

        col_reg_m, col_reg_chart = st.columns([0.45, 0.55])

        with col_reg_m:
            st.markdown("##### 📊 Regression Performance Metrics")
            base_reg = reg_metrics.get("zero_return_baseline", {})
            reg_table = [
                {"Model": "Linear Regression", "MAE": f"{lr_reg.get('mae', 0.0):.6f}", "RMSE": f"{lr_reg.get('rmse', 0.0):.6f}", "R² Score": f"{lr_reg.get('r2', 0.0):.4f}"},
                {"Model": "Random Forest Regressor", "MAE": f"{rf_reg.get('mae', 0.0):.6f}", "RMSE": f"{rf_reg.get('rmse', 0.0):.6f}", "R² Score": f"{rf_reg.get('r2', 0.0):.4f}"},
                {"Model": "Zero-Return Baseline", "MAE": f"{base_reg.get('mae', 0.0):.6f}", "RMSE": f"{base_reg.get('rmse', 0.0):.6f}", "R² Score": f"{base_reg.get('r2', 0.0):.4f}"},
            ]
            st.dataframe(reg_table, use_container_width=True, hide_index=True)

            st.info(
                "💡 Financial market daily returns exhibit high stochasticity and near-martingale properties. "
                "Negative R² is standard in raw equity returns, demonstrating that market direction classifiers "
                "outperform point-return regressors."
            )

        with col_reg_chart:
            reg_preds = svc.get_regression_predictions()
            if not reg_preds.empty and "actual" in reg_preds.columns:
                fig_reg = go.Figure()
                sample_preds = reg_preds.tail(100)
                fig_reg.add_trace(go.Scatter(x=sample_preds["date"], y=sample_preds["actual"], mode="lines", name="Actual Return", line=dict(color="#2962FF", width=1.5)))
                if "predicted_linear" in sample_preds.columns:
                    fig_reg.add_trace(go.Scatter(x=sample_preds["date"], y=sample_preds["predicted_linear"], mode="lines", name="Linear Pred", line=dict(color="#FFB300", width=1.5, dash="dash")))
                if "predicted_rf" in sample_preds.columns:
                    fig_reg.add_trace(go.Scatter(x=sample_preds["date"], y=sample_preds["predicted_rf"], mode="lines", name="Random Forest Pred", line=dict(color="#00C805", width=1.5, dash="dot")))
                fig_reg.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#161922",
                    title="<b>Actual vs. Predicted Return (Last 100 Test Samples)</b>",
                    height=340,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig_reg, use_container_width=True)

        if not reg_preds.empty:
            render_downloadable_table(
                df=reg_preds,
                title="Test-Set Out-of-Sample Regression Predictions",
                filename="regression_test_predictions.csv",
                height=260,
            )

    # --- TAB 3: CLUSTERING ---
    with tab_cluster:
        st.markdown("#### 🧩 Stock Risk-Return Clustering (K-Means)")
        st.caption("Clustering 10 Indian equities by mean daily return, annualized volatility, trading volume, and maximum drawdown.")

        clusters_df = svc.get_stock_clusters()
        if not clusters_df.empty:
            col_cl_chart, col_cl_profiles = st.columns([0.55, 0.45])

            with col_cl_chart:
                fig_cl = create_cluster_scatter(clusters_df)
                st.plotly_chart(fig_cl, use_container_width=True)

            with col_cl_profiles:
                st.markdown("##### 📑 Cluster Profiles & Characterization")
                profiles = cluster_metrics.get("cluster_profiles", [])
                for p in profiles:
                    cid = p.get("cluster")
                    ret = p.get("avg_daily_return", 0.0) * 100
                    vol = p.get("avg_volatility", 0.0) * 100
                    dd = p.get("max_drawdown", 0.0) * 100
                    syms = p.get("symbols", "")

                    st.markdown(
                        f"""
                        <div style="background-color: #1E222D; border-left: 3px solid #2962FF; padding: 10px 14px; margin-bottom: 10px; border-radius: 4px; font-size: 0.8rem;">
                            <div style="font-weight: 700; color: #FFFFFF;">Cluster {cid}: {p.get('stock_count')} Equities</div>
                            <div style="color: #A0A5B5; margin-top: 2px;"><strong>Stocks:</strong> {syms}</div>
                            <div style="color: #787B86; margin-top: 4px;">
                                Return: <span style="color: {'#00C805' if ret>=0 else '#FF3B30'}">{ret:+.3f}%</span> |
                                Volatility: <strong>{vol:.2f}%</strong> |
                                Max Drawdown: <strong>{dd:.1f}%</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            render_downloadable_table(
                df=clusters_df,
                title="K-Means Stock Cluster Assignments & Metrics",
                filename="stock_clusters.csv",
                height=260,
            )

    # --- TAB 4: INTERACTIVE INFERENCE ---
    with tab_infer:
        st.markdown("#### 🔮 Live Market Direction Inference Simulation")
        st.caption("Test the trained Random Forest classifier with custom technical indicators in real-time.")

        st.markdown(
            """
            <div style="background-color: #211915; border-left: 3px solid #FF9800; padding: 10px 16px; border-radius: 4px; font-size: 0.8rem; color: #FFA726; margin-bottom: 20px;">
                <strong>NOTICE:</strong> This simulation invokes the serialized <code>classification_random_forest.joblib</code> estimator
                on custom inputs for educational validation. Outputs are statistical class likelihoods.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            in_ret = st.number_input("Daily Return (e.g. 0.01 = +1%)", value=0.005, step=0.001, format="%.4f")
            in_vol = st.number_input("20-Day Volatility (e.g. 0.015)", value=0.015, step=0.001, format="%.4f")
        with col_in2:
            in_vol_chg = st.number_input("Volume Change % (e.g. 0.1 = +10%)", value=0.05, step=0.01, format="%.4f")
            in_rsi = st.slider("RSI 14 (Relative Strength)", min_value=0.0, max_value=100.0, value=58.0, step=0.5)
        with col_in3:
            in_macd = st.number_input("MACD", value=1.25, step=0.1, format="%.2f")
            in_macd_sig = st.number_input("MACD Signal", value=0.85, step=0.1, format="%.2f")

        if st.button("🚀 Run Random Forest Direction Inference", use_container_width=True):
            input_features = {
                "daily_return": in_ret,
                "volatility": in_vol,
                "volume_change": in_vol_chg,
                "rsi_14": in_rsi,
                "macd": in_macd,
                "macd_signal": in_macd_sig,
            }
            result = svc.predict_market_direction(input_features)

            if "error" in result:
                st.error(result["error"])
            else:
                pred = result["prediction"]
                conf = result["confidence"] * 100
                probs = result["probabilities"]

                pred_color = "#00C805" if pred == "Bullish" else ("#FF3B30" if pred == "Bearish" else "#FFB300")
                st.markdown(
                    f"""
                    <div class="terminal-card" style="text-align: center; border-color: {pred_color};">
                        <div style="font-size: 0.9rem; color: #787B86; text-transform: uppercase;">Predicted Market Direction</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: {pred_color}; margin: 8px 0;">{pred.upper()}</div>
                        <div style="font-size: 1rem; color: #FFFFFF;">Model Confidence: <strong>{conf:.1f}%</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("##### Class Probability Distribution")
                prob_df = [{"Class": k, "Probability": f"{v*100:.2f}%"} for k, v in probs.items()]
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

    render_disclaimer()


if __name__ == "__main__":
    main()

