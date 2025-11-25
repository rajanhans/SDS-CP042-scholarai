"""Simple, easy-to-read multi-agent orchestrator.

This module builds one main orchestration function `analyze_stock` that:
- creates three helper agents: `current_price_agent`, `sentiment_analysis_agent`,
  and `technical_analysis_agent`;
- runs the three helpers in parallel (via threads);
- then feeds their textual outputs to `synthesizer_agent` to produce a single
  'Buy' / 'Hold' / 'Sell' recommendation with reasoning.

Usage (CLI):
    python Initial_research_agent.py AAPL --model gpt-4.1-mini

The code intentionally keeps the prompts and structure minimal so it's easy
to read and modify.
"""

import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

from agents import Agent, Runner

load_dotenv()
DEFAULT_MODEL = "gpt-4.1-mini"



def build_agents(model_name: str = DEFAULT_MODEL) -> Tuple[Agent, Agent, Agent, Agent]:
    """Create and return the helper agents and the synthesizer.

    Returns: (current_price_agent, sentiment_analysis_agent,
              technical_analysis_agent, synthesizer_agent)
    """

    current_price_agent = Agent(
        name="current_price_agent",
        model=model_name,
        instructions=(
            "You are a market-data researcher. Given a stock ticker, do a short, focused"
            " investigation and return the result in three parts:\n"
            "1) Price Summary (1–2 sentences): current approximate price band, recent"
            " intraday/24h movement, and short-term trend (up/down/sideways).\n"
            "2) Key Recent Drivers (3 bullet points): recent news/events, earnings,"
            " or macro factors that could explain the move. Use dates where relevant.\n"
            "3) Confidence & Sources (1 line): a short confidence tag (High/Medium/Low)"
            " and 1–2 short source hints (e.g., 'news headlines', 'earnings release',"
            " 'market data').\n"
            "Keep answers concise and factual. If you are unsure about exact numeric"
            " price, state uncertainty explicitly and provide a reasoned range."
        ),
    )

    sentiment_analysis_agent = Agent(
        name="sentiment_analysis_agent",
        model=model_name,
        instructions=(
            "You are a sentiment researcher. For the given ticker, perform a concise"
            " sentiment review and return three sections:\n"
            "1) Headline Sentiment (1 sentence): overall tone (Positive/Neutral/Negative).\n"
            "2) Supporting Evidence (3 short bullets): recent headlines, social-media"
            " signals or analyst notes that justify the headline sentiment. Include dates"
            " or short citations where possible.\n"
            "3) Market Impact & Confidence (1 line): expected near-term impact (Bullish/"
            "Bearish/Neutral) and a confidence tag (High/Medium/Low).\n"
            "Prefer factual statements and brief citations; avoid speculation beyond the"
            " evidence you list."
        ),
    )

    technical_analysis_agent = Agent(
        name="technical_analysis_agent",
        model=model_name,
        instructions=(
            "You are a technical analyst. For the given ticker, produce three sections:\n"
            "1) Trend Summary (1 sentence): short/medium-term trend (rising/falling/sideways).\n"
            "2) Indicators & Levels (3 bullets): describe likely state of key indicators"
            " (moving averages, RSI, MACD in words) and list 1–2 approximate support/resistance"
            " ranges (rounded).\n"
            "3) Trade Context & Confidence (1 line): what kind of trader/view this suits"
            " (short-term swing, longer-term investor) and a confidence tag.\n"
            "If precise indicator values aren't available, describe expected direction and"
            " why (e.g., 'price below 50-day MA -> bearish momentum'). Keep answers concise."
        ),
    )

    synthesizer_agent = Agent(
        name="synthesizer_agent",
        model=model_name,
        instructions=(
            "You are an investment synthesizer. You receive three labeled sections"
            " with structured sub-sections from specialist agents.\n\n"
            "Input Format:\n"
            "- Price: includes 'Price Summary', 'Key Recent Drivers' (3 bullets),"
            " 'Confidence & Sources'.\n"
            "- Sentiment: includes 'Headline Sentiment', 'Supporting Evidence' (3 bullets),"
            " 'Market Impact & Confidence'.\n"
            "- Technical: includes 'Trend Summary', 'Indicators & Levels' (3 bullets),"
            " 'Trade Context & Confidence'.\n\n"
            "Your Task:\n"
            "1) Read each section carefully, noting confidence tags and sources.\n"
            "2) Produce exactly ONE recommendation: 'Buy', 'Hold', or 'Sell'.\n"
            "3) Explain your reasoning with:\n"
            "   - 3–4 key supporting points (cite evidence from the three sections).\n"
            "   - 1 main risk or invalidation scenario.\n"
            "   - Timeframe for your view (e.g., 'short-term swing', 'intermediate hold').\n"
            "4) Use confidence levels from the specialist agents to gauge conviction.\n"
            "Prefer evidence-based reasoning; flag any conflicts or low-confidence areas."
        ),
    )

    return (
        current_price_agent,
        sentiment_analysis_agent,
        technical_analysis_agent,
        synthesizer_agent,
    )


def _run_agent_and_get_text(agent: Agent, prompt: str) -> str:
    """Run an agent synchronously and return the textual output.

    Uses `Runner.run_sync` and falls back gracefully if the returned object
    doesn't expose `final_output`.
    """
    res = Runner.run_sync(agent, prompt)
    return getattr(res, "final_output", str(res))


def analyze_stock(ticker: str, model_name: str = DEFAULT_MODEL) -> str:
    """Orchestrate helpers for `ticker` and return the synthesizer output.

    Steps:
    1. Build agents with `build_agents`.
    2. Create small prompts for each helper.
    3. Run the three helpers concurrently using threads.
    4. Combine their outputs and call the synthesizer.
    """
    (price_agent, sentiment_agent, technical_agent, synth_agent) = build_agents(
        model_name
    )

    # Small, specific prompts for each helper
    price_prompt = f"Ticker: {ticker}\nProvide a short price context (1-2 sentences)."
    sentiment_prompt = (
        f"Ticker: {ticker}\nProvide 5 short bullets about market sentiment."
    )
    technical_prompt = (
        f"Ticker: {ticker}\nProvide a concise technical summary (trend + 1-2 levels). Do not miss any important details"
    )

    helpers = [
        ("Price", price_agent, price_prompt),
        ("Sentiment", sentiment_agent, sentiment_prompt),
        ("Technical", technical_agent, technical_prompt),
    ]

    results = {}
    # Run helpers in parallel threads to keep the code simple but concurrent
    with ThreadPoolExecutor(max_workers=3) as exe:
        futures = {
            exe.submit(_run_agent_and_get_text, a, p): label
            for (label, a, p) in helpers
        }
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                results[label] = fut.result()
            except Exception as e:
                results[label] = f"ERROR: {e}"

    # Build synthesizer prompt by labeling each section clearly
    synth_input = (
        f"Ticker: {ticker}\n\n"
        "Price:\n" + results.get("Price", "(no data)") + "\n\n"
        "Sentiment:\n" + results.get("Sentiment", "(no data)") + "\n\n"
        "Technical:\n" + results.get("Technical", "(no data)") + "\n\n"
        "Based on the sections above, give exactly one recommendation (Buy/Hold/Sell),"
        " 3-5 supporting points, and one risk that could invalidate the view."
    )

    final = _run_agent_and_get_text(synth_agent, synth_input)
    return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple stock research orchestrator")
    parser.add_argument(
        "ticker", nargs="?", default="AAPL", help="Ticker symbol (e.g. AAPL)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model name to use for agents"
    )
    args = parser.parse_args()

    output = analyze_stock(args.ticker, model_name=args.model)
    print("\n=== FINAL RECOMMENDATION ===\n")
    print(output)
