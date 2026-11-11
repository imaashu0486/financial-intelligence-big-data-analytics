"""
Page 5: Association Rule Mining.
Visualizes Apriori basket analysis for equity co-movement, frequent itemsets,
and association rules with support, confidence, and lift metrics.
"""

from pathlib import Path
import sys
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.layout import setup_page, render_terminal_header, render_disclaimer
from app.components.sidebar import render_sidebar
from app.components.metrics import render_kpi_grid
from app.components.charts import create_apriori_scatter
from app.components.tables import render_downloadable_table
from app.services.mining_service import MiningService


def main() -> None:
    setup_page("Association Rules", icon="🔗")
    render_sidebar(active_page="Association Rules")

    render_terminal_header(
        title="APRIORI MARKET CO-MOVEMENT & ASSOCIATION RULES",
        subtitle="TRANSACTION BASKET MINING, FREQUENT ITEMSETS & LIFT METRICS (PHASE 6)",
        badge="UNSUPERVISED MINING",
    )

    svc = MiningService()
    stats = svc.get_mining_statistics()
    tx_summary = svc.get_transaction_summary()

    if not stats:
        st.error("Mining statistics not found in data/processed/association/association_statistics.json.")
        return

    # Top KPI Metrics Row
    total_tx = tx_summary.get("num_transactions") or tx_summary.get("total_transactions", 0)
    unique_items = tx_summary.get("num_unique_items") or (len(tx_summary.get("unique_items", [])) if isinstance(tx_summary.get("unique_items"), list) else 0)
    frequent_itemsets = stats.get("num_frequent_itemsets") or stats.get("total_frequent_itemsets", 0)
    discovered_rules = stats.get("num_rules") or stats.get("total_rules_generated", 0)
    max_lift = stats.get("rule_metrics", {}).get("lift", {}).get("max") or stats.get("max_lift", 0.0)

    kpis = [
        {"label": "Total Daily Transactions", "value": f"{total_tx:,}", "delta": "Daily Market Baskets"},
        {"label": "Unique Event Items", "value": f"{unique_items} Items", "delta": "10 Equities × {UP}"},
        {"label": "Frequent Itemsets", "value": f"{frequent_itemsets} Itemsets", "delta": "Support ≥ 0.05"},
        {"label": "Discovered Rules", "value": f"{discovered_rules} Rules", "delta": "Confidence ≥ 0.30"},
        {"label": "Maximum Rule Lift", "value": f"{max_lift:.4f}", "delta": "Strong Positive Co-Movement"},
    ]
    render_kpi_grid(kpis, columns=5)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filtering Sliders
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        min_supp = st.slider("Minimum Support Threshold:", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    with col_f2:
        min_conf = st.slider("Minimum Confidence Threshold:", min_value=0.10, max_value=0.80, value=0.30, step=0.05)
    with col_f3:
        min_lift = st.slider("Minimum Lift Threshold:", min_value=1.00, max_value=2.00, value=1.10, step=0.05)

    rules_df = svc.get_association_rules(
        min_support=min_supp,
        min_confidence=min_conf,
        min_lift=min_lift,
    )

    # Interactive Scatter Visualization
    st.markdown("#### 🌌 Mined Rule Landscape (Support vs. Confidence)")
    st.caption("Bubble size and color represent Lift: Lift > 1 indicates that the antecedent and consequent occur together more frequently than expected by chance.")

    if not rules_df.empty:
        fig_scatter = create_apriori_scatter(rules_df)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No association rules meet the specified filtering thresholds. Try lowering the support or lift sliders.")

    # Tabs for Rules and Itemsets
    tab_rules, tab_itemsets = st.tabs(["📜 Filtered Association Rules", "📦 Discovered Frequent Itemsets"])

    with tab_rules:
        if not rules_df.empty:
            # Add formatted rule string
            display_rules = rules_df.copy()
            display_rules["Rule Formulation"] = display_rules["antecedent"] + " ➔ " + display_rules["consequent"]
            cols = ["Rule Formulation", "support", "confidence", "lift", "occurrence_count"]
            cols_avail = [c for c in cols if c in display_rules.columns]
            table_rules = display_rules[cols_avail].copy()

            render_downloadable_table(
                df=table_rules,
                title="Discovered Market Co-Movement Rules",
                filename="filtered_association_rules.csv",
                height=320,
            )
        else:
            st.info("No rules to display.")

    with tab_itemsets:
        itemsets_df = svc.get_frequent_itemsets(min_support=min_supp)
        if not itemsets_df.empty:
            col_it1, col_it2 = st.columns([0.45, 0.55])
            with col_it1:
                st.markdown("##### 🏆 Top Frequent Itemsets by Support")
                st.dataframe(itemsets_df.head(20), use_container_width=True, hide_index=True)

            with col_it2:
                st.markdown("##### 📊 Top Itemset Support Distribution")
                top_15 = itemsets_df.head(15)
                fig_it = px.bar(
                    top_15,
                    x="support",
                    y="itemset",
                    orientation="h",
                    color="support",
                    color_continuous_scale="Purp",
                    template="plotly_dark",
                )
                fig_it.update_layout(
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#161922",
                    yaxis=dict(autorange="reversed"),
                    height=360,
                )
                st.plotly_chart(fig_it, use_container_width=True)
        else:
            st.info("No frequent itemsets found with current support threshold.")

    render_disclaimer()


if __name__ == "__main__":
    main()

