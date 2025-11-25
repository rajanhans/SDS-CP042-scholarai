"""Streamlit app to invoke the Initial_research_agent analyze_stock workflow.

Usage:
    pip install streamlit
    streamlit run streamlit_app.py

The app lets you choose a ticker and a model name and runs the analysis.
"""

import streamlit as st

from Initial_research_agent import analyze_stock, DEFAULT_MODEL

st.set_page_config(page_title="Stock Research Demo", layout="centered")

st.title("Stock Research Orchestrator")

col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()
with col2:
    model_name = st.text_input("Model name", value=DEFAULT_MODEL)

run = st.button("Run Analysis")

if run:
    if not ticker:
        st.error("Please enter a ticker symbol.")
    else:
        with st.spinner("Running research agents — this may take a few seconds..."):
            try:
                result = analyze_stock(ticker, model_name=model_name)
                st.subheader("Final recommendation")
                st.write(result)
            except Exception as e:
                st.error(f"Error running analysis: {e}")
                raise
