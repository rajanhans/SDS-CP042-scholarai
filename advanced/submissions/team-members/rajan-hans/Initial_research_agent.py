"""
Very simple multi-agent stock workflow using OpenAI Agents SDK + handoffs.

Agents:
- research_agent: plans & delegates via handoffs
- current_price_agent: describes current price context for the ticker
- sentiment_analysis_agent: analyzes news / sentiment
- technical_analysis_agent: provides simple technical analysis
- synthesizer_agent: merges everything into Buy/Hold/Sell

NOTE: This is a *skeleton* — helper agents just reason from model knowledge.
Replace their instructions/tools with real data sources as you evolve it.
"""

import os
from dotenv import load_dotenv
from agents import Agent, Runner, ModelSettings

DEFAULT_MODEL = "gpt-4.1-mini"  # or any model you prefer
print("Program started")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Helper agents
# ---------------------------------------------------------------------------

current_price_agent = Agent(
    name="current_price_agent",
    model=DEFAULT_MODEL,
    instructions=(
        "You are a stock price specialist.\n"
        "Task: Given a stock ticker symbol in the conversation, "
        "describe the *approximate* current price zone and recent price action.\n\n"
        "Constraints:\n"
        "- If you are not sure of the exact live price, say so explicitly.\n"
        "- Still provide a rough price range and short-term price trend "
        "(uptrend, downtrend, sideways) based on your knowledge.\n"
        "- Keep the answer focused on price, support/resistance, and volatility.\n"
    ),
)

sentiment_analysis_agent = Agent(
    name="sentiment_analysis_agent",
    model=DEFAULT_MODEL,
    instructions=(
        "You are a market sentiment analyst.\n"
        "Task: For the given stock ticker, analyze overall market sentiment.\n\n"
        "Consider:\n"
        "- Recent news tone (positive/neutral/negative)\n"
        "- Social media and retail sentiment (bullish/bearish/mixed)\n"
        "- Any notable macro or sector mood that affects this ticker\n\n"
        "Important:\n"
        "- If you are not sure about very recent news, say so explicitly.\n"
        "- Still provide a reasoned sentiment view and list 3–5 key points.\n"
    ),
)

technical_analysis_agent = Agent(
    name="technical_analysis_agent",
    model=DEFAULT_MODEL,
    instructions=(
        "You are a technical analysis specialist.\n"
        "Task: For the given stock ticker, provide a concise technical view.\n\n"
        "Base your answer on:\n"
        "- Trend (short-, medium-, and long-term if possible)\n"
        "- Key indicators in words (e.g., moving averages, RSI, MACD — "
        "you can describe their likely state even if you don't have exact values).\n"
        "- Important support and resistance levels (approximate ranges are fine).\n"
        "- Momentum and volatility assessment.\n"
        "Be explicit about any uncertainty and keep the answer focused on TA only.\n"
    ),
)

synthesizer_agent = Agent(
    name="synthesizer_agent",
    model=DEFAULT_MODEL,
    instructions=(
        "You are an investment recommendation synthesizer.\n"
        "You receive a conversation that already includes:\n"
        "- A research plan from the research agent\n"
        "- Findings from current_price_agent\n"
        "- Findings from sentiment_analysis_agent\n"
        "- Findings from technical_analysis_agent\n\n"
        "Your job:\n"
        "1. Read all previous content carefully.\n"
        "2. Produce a clear recommendation for the stock: exactly one of 'Buy', 'Hold', or 'Sell'.\n"
        "3. Explain your reasoning in detail.\n"
        "4. List key supporting points from:\n"
        "   - Price & recent trend\n"
        "   - Sentiment & news\n"
        "   - Technical indicators / chart structure\n"
        "5. Outline the main risks and what could invalidate your view.\n"
        "6. Suggest practical next steps or checks (e.g., confirm latest earnings, "
        "check news, validate levels on a real chart).\n"
    ),
)

# ---------------------------------------------------------------------------
# Main research / orchestration agent (uses handoffs)
# ---------------------------------------------------------------------------

research_agent = Agent(
    name="research_agent",
    model=DEFAULT_MODEL,
    # Important: parallel_tool_calls=True allows the model to call multiple
    # handoffs (sub-agents) in the same turn if it decides to.
    model_settings=ModelSettings(
        temperature=0.3,
        parallel_tool_calls=True,
    ),
    instructions=(
        "You are the main research coordinator for stock analysis.\n\n"
        "Overall task:\n"
        "- Given a stock ticker, create a brief research plan.\n"
        "- Then delegate work to specialized agents via handoffs:\n"
        "  * current_price_agent\n"
        "  * sentiment_analysis_agent\n"
        "  * technical_analysis_agent\n"
        "- After helpers have done their work, hand off to synthesizer_agent "
        "to produce the final 'Buy', 'Hold', or 'Sell' recommendation.\n\n"
        "Process you MUST follow:\n"
        "1. **Planning step** – First, write a short 'Research Plan' as a numbered list "
        "   (1–5 steps max) tailored to the given ticker.\n"
        "2. **Execution step** – Use handoffs to the helper agents listed above. "
        "   You may call them in any order; it is okay to use them in parallel.\n"
        "3. **Synthesis step** – Once you judge that the helpers have provided enough "
        "   information, hand off control to the synthesizer_agent so it can make "
        "   the final investment recommendation.\n\n"
        "Important details:\n"
        "- Do not try to be the final decision-maker yourself.\n"
        "- Your role ends once you hand off to synthesizer_agent.\n"
        "- Make sure each helper agent clearly sees the stock ticker in the conversation.\n"
    ),
    # Handoffs: these agents can be delegated to by the research_agent.
    handoffs=[
        current_price_agent,
        sentiment_analysis_agent,
        technical_analysis_agent,
        synthesizer_agent,
    ],
)


# ---------------------------------------------------------------------------
# Small helper to run the workflow for a given ticker
# ---------------------------------------------------------------------------


def analyze_stock(ticker: str) -> str:
    """
    Kicks off the whole workflow starting from the research_agent.

    The research_agent will:
      - Plan
      - Handoff to helpers
      - Handoff to synthesizer_agent

    Returns the final text output (synthesizer_agent's recommendation).
    """
    user_task = (
        f"Analyze stock '{ticker}'.\n"
        f"Follow your planning → helper handoffs → synthesizer handoff workflow.\n"
        f"Final result should be a clear 'Buy', 'Hold', or 'Sell' with reasoning."
    )

    result = Runner.run_sync(research_agent, user_task)
    # result.final_output is the final string produced by the last agent (synthesizer)
    return result.final_output


if __name__ == "__main__":
    ticker = "AAPL"
    final_report = analyze_stock(ticker)
    print("\n=== FINAL RECOMMENDATION ===\n")
    print(final_report)
