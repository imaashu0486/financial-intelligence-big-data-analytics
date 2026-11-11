"""
Real-Data News and Corporate Document Intelligence Page (Step 9 Phase C).
Displays 2024 news disclosure NLP, return co-movement analysis, and FY24 corporate filing extracts.
"""

import json
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
from app.services.analytics_service import AnalyticsService


def main() -> None:
    setup_page("Real News & Filings", icon="🗞️")
    render_sidebar(active_page="Real News & Filings")

    render_terminal_header(
        title="NEWS INTELLIGENCE & CORPORATE DOCUMENT ANALYSIS",
        subtitle="2024 GDELT DISCLOSURES, FINANCIAL SENTIMENT & FY24 MD&A FILING EXTRACTS",
        badge="REAL TEXT ASSETS // ACADEMIC NON-CAUSALITY BOUNDARY",
    )

    analytics_svc = AnalyticsService()
    news_report = analytics_svc.get_real_news_intelligence()

    if not news_report:
        st.warning("News intelligence report not found. Please verify Step 8 outputs.")
        return

    scope = news_report.get("dataset_scope", {})
    categories = news_report.get("event_category_distribution", {})
    sent_dist = news_report.get("sentiment_distribution", {})
    assoc = news_report.get("market_return_temporal_association", {})
    docs = news_report.get("corporate_filings_extract_summary", [])
    company_summary = news_report.get("company_disclosure_summary", [])

    # Top KPI Metrics
    metric_cards = [
        {
            "label": "2024 Disclosures Analyzed",
            "value": f"{scope.get('total_news_records', 79)} Articles",
            "delta": f"{scope.get('instrument_coverage_count', 32)} Equities Covered",
            "help": "Official news disclosures collected via GDELT across calendar year 2024.",
        },
        {
            "label": "Mean Corpus Sentiment",
            "value": f"+{sent_dist.get('mean_score', 0.2862):.3f}",
            "delta": "Financial Lexicon Scale [-1, +1]",
            "help": "Mean score computed by Loughran-McDonald adapted financial lexicon.",
        },
        {
            "label": "Same-Day Return Correlation",
            "value": f"r = {assoc.get('same_day_correlation', 0.0888):+.4f}",
            "delta": "p = 0.4751 (N=67)",
            "help": "Contemporaneous co-movement between news sentiment and disclosure-day return.",
        },
        {
            "label": "Corporate Filing Extracts",
            "value": f"{scope.get('corporate_documents_analyzed', 4)} Companies",
            "delta": "FY2023–24 Annual Reports",
            "help": "Executive MD&A extracts for Reliance, TCS, HDFC Bank, and Infosys.",
        },
    ]
    render_kpi_grid(metric_cards, columns=4)

    st.markdown("<br>", unsafe_allow_html=True)

    # Formal Academic Statistical Conclusion Banner
    st.info(
        "**Statistical Interpretation Standard (Step 8.1.1 Guidance)**:\n\n"
        "\"No statistically significant linear association was detected between the available news sentiment observations "
        "and same-day or next-day returns at the 5% significance level. The results do not establish the absence of an association, "
        "nor do they establish causality or predictive power.\""
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for News vs Corporate Filings
    tab_categories, tab_companies, tab_filings, tab_limitations = st.tabs([
        "🏷️ Disclosure Categories & Sentiment",
        "🏢 Company Disclosure Activity",
        "📑 Corporate Annual Report Extracts (FY24)",
        "⚠️ Scope & Academic Boundaries",
    ])

    with tab_categories:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Event Category Breakdown")
            df_cat = pd.DataFrame(list(categories.items()), columns=["Event Category", "Article Count"])
            fig_cat = px.pie(
                df_cat, names="Event Category", values="Article Count",
                title="2024 Disclosure Category Distribution (N=79)",
                hole=0.4, color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig_cat.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig_cat, use_container_width=True)

        with c2:
            st.subheader("Financial Sentiment Distribution")
            df_sent = pd.DataFrame([
                {"Label": "Positive", "Count": sent_dist.get("Positive", 38), "Color": "#00C805"},
                {"Label": "Neutral", "Count": sent_dist.get("Neutral", 38), "Color": "#2962FF"},
                {"Label": "Negative", "Count": sent_dist.get("Negative", 3), "Color": "#FF3B30"},
            ])
            fig_sent = px.bar(
                df_sent, x="Label", y="Count", color="Label",
                color_discrete_map={"Positive": "#00C805", "Neutral": "#2962FF", "Negative": "#FF3B30"},
                title="Sentiment Class Distribution (Mean Score: +0.2862)",
            )
            fig_sent.update_layout(height=400, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_sent, use_container_width=True)

    with tab_companies:
        st.subheader("Company-Wise Disclosure Frequency & Average Sentiment")
        if company_summary:
            df_comp = pd.DataFrame(company_summary)
            df_disp_comp = df_comp.copy()
            df_disp_comp["mean_sentiment"] = df_disp_comp["mean_sentiment"].round(3)
            st.dataframe(df_disp_comp, use_container_width=True, height=450)

    with tab_filings:
        st.subheader("Executive Management Discussion & Analysis (MD&A) Extracts")
        st.markdown(
            "*Authentic strategic overview extracts from FY24 annual reports. Note: These are concise executive extracts, not complete 400-page regulatory filings.*"
        )
        for doc in docs:
            with st.expander(f"📄 {doc.get('filename')} (Words: {doc.get('word_count'):,}, Characters: {doc.get('character_count'):,})"):
                st.write(f"**Identified Strategic Themes**: {', '.join(doc.get('key_themes', []))}")
                raw_path = ROOT / "data" / "real" / "raw" / "documents" / doc.get("filename")
                if raw_path.exists():
                    text_content = raw_path.read_text(encoding="utf-8", errors="ignore")
                    st.text_area("Extract Content Preview:", text_content, height=220)

    with tab_limitations:
        st.subheader("Corpus Scope & Explicit Academic Limitations")
        st.markdown(
            """
            1. **Cross-Sectional 2024 Window**: The news disclosure dataset spans calendar year 2024 ($N=79$) and does not represent an exhaustive 10-year tick feed.
            2. **Official Press Releases**: Data comprises official lead headlines and regulatory disclosure summaries, creating an inherent corporate positive bias (+0.2862 mean score).
            3. **Executive Document Extracts**: The corporate filing documents are focused MD&A and strategic overview extracts for 4 large caps, not complete statutory annual accounts.
            4. **Absence of Causality**: The failure to detect significant linear return predictability ($p > 0.05$) is consistent with efficient financial markets.
            """
        )

    render_disclaimer()


if __name__ == "__main__":
    main()

