import os

from agents import Agent, Runner, WebSearchTool, InputGuardrail, GuardrailFunctionOutput
from agents.exceptions import InputGuardrailTripwireTriggered
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio


print("Program started")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("Open API key is " + OPENAI_API_KEY)

main_research_agent = Agent(
    name="Main Stock Research Agent",
    instructions="You are the main stock research assistant. You will invoke the other research helper agents as needed.",
    model="gpt-4o-mini",
)

market_trend_agent = Agent(
    name="Stock Market Trend Agent",
    instructions="You are a stock market trend assistant. Provide the open and close price of the stock in question for the previous day.",
    model="gpt-4o-mini",
)

sentiment_analysis_agent = Agent(
    name="Stock Sentiment Analysis Agent",
    instructions="You are a stock sentiment analysis assistant. Provide insights on the market sentiment based on recent news and social media.",
    model="gpt-4o-mini",
)

technical_analysis_agent = Agent(
    name="Technical Analysis Agent",
    instructions="You are a technical analysis assistant. Provide technical indicators and analysis for the stock in question.",
    model="gpt-4o-mini",
)


synthesizer_agent = Agent(
    name="Synthesizer Agent",
    instructions=(
        "You are a synthesizer and investment recommendation assistant. "
        "Given combined findings from market trends, sentiment, and technical analysis, provide a clear recommendation: 'Buy', 'Hold', or 'Sell'. "
        "Explain your reasoning in detail, list key supporting points, outline risks, and suggest next steps or checks to perform before taking action."
    ),
    model="gpt-4o-mini",
)


async def safe_run(agent, prompt, retries=2):
    """Run an agent with a small retry/backoff policy and return the result."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return await Runner.run(agent, prompt)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(1 + attempt)
            else:
                raise


async def main():
    # 1) Ask the main research agent for a plan
    task_prompt = "Research the stock TSLA for today's close: which helper agents should we invoke and what's the question to each?"
    plan = await safe_run(main_research_agent, task_prompt)
    print("Main agent plan:\n", plan)

    # 2) Run the sub-agents and collect results concurrently
    try:
        candle_prompt = (
            "For TSLA, provide previous trading day's open and close prices."
        )
        sentiment_prompt = "For TSLA, summarize market sentiment from recent news and social media (short)."
        technical_prompt = "For TSLA, compute key technical indicators (e.g. 20/50 MA, RSI) and give a short interpretation."

        # create tasks (do not await here so they run concurrently)
        candle_task = safe_run(candle_stick_agent, candle_prompt)
        sentiment_task = safe_run(sentiment_analysis_agent, sentiment_prompt)
        technical_task = safe_run(technical_analysis_agent, technical_prompt)

        # await all tasks concurrently; if any raises, the exception will propagate
        candle_result, sentiment_result, technical_result = await asyncio.gather(
            candle_task, sentiment_task, technical_task
        )

        print("Candle result:\n", candle_result)
        print("Sentiment result:\n", sentiment_result)
        print("Technical result:\n", technical_result)

        # 3) Combine findings and send to the synthesizer agent for a buy/hold/sell recommendation
        combined = (
            "CANDLE:\n" + str(candle_result) + "\n\n"
            "SENTIMENT:\n" + str(sentiment_result) + "\n\n"
            "TECHNICAL:\n" + str(technical_result)
        )
        synth_prompt = (
            "Here are the combined research findings for a stock. Based on these, provide a clear recommendation (Buy/Hold/Sell) and a detailed rationale. "
            "Include supporting points, key risks, confidence level, and suggested next steps.\n\n"
            + combined
        )
        synthesizer_result = await safe_run(synthesizer_agent, synth_prompt)
        print("Synthesizer recommendation:\n", synthesizer_result)

    except Exception as e:
        print("Error running sub-agents:", e)


if __name__ == "__main__":
    asyncio.run(main())
