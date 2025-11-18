import os

from agents import Agent, Runner, WebSearchTool, InputGuardrail, GuardrailFunctionOutput
from agents.exceptions import InputGuardrailTripwireTriggered
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
import json
import re
from typing import Any
from research_tools import YahooPriceTool


print("Program started")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# print("Open API key is " + OPENAI_API_KEY)

""" ORGANIZATION OF AGENTS:
- A top-level orchestrator agent ✅
- Multiple specialized agents ✅
- Concurrent execution of sub-agents ✅
- A synthesizer that makes a final decision ✅
"""

"""  SPECIALIZED WORKER AGENTS:
- weekly_price_agent → historical weekly highs/lows
- sentiment_analysis_agent → news/social sentiment
- technical_analysis_agent → indicators/interpretation
- synthesizer_agent → combine into Buy/Hold/Sell 
"""
current_price_agent = Agent(
    name="Stock Market current Price Agent",
    instructions="You are stock mkt pricing assistant. Provide the current price of the stock in question. ",
    model="gpt-4o-mini",
    handoff_description="You will provide the current price for the stock in question.",
)

# Instantiate price tool (async wrapper)
price_tool = YahooPriceTool()

sentiment_analysis_agent = Agent(
    name="Stock Sentiment Analysis Agent",
    instructions="You are a stock sentiment analysis assistant. Provide insights on the market sentiment based on recent news and social media.",
    model="gpt-4o-mini",
    handoff_description="You will provide a summary of market sentiment based on recent news and social media.",
)

technical_analysis_agent = Agent(
    name="Technical Analysis Agent",
    instructions="You are a technical analysis assistant. Provide technical indicators and analysis for the stock in question.",
    model="gpt-4o-mini",
    handoff_description="You will provide technical indicators and analysis for the stock in question.",
)

synthesizer_agent = Agent(
    name="Synthesizer Agent",
    instructions=(
        "You are a synthesizer and investment recommendation assistant. "
        "Given combined findings from market trends, sentiment, and technical analysis, provide a clear recommendation: 'Buy', 'Hold', or 'Sell'. "
        "Explain your reasoning in detail, list key supporting points, outline risks, and suggest next steps or checks to perform before taking action."
    ),
    model="gpt-4o-mini",
    handoff_description="You will provide a synthesized investment recommendation based on combined findings.",
)

"""DEFINE A PYDANTIC SCHEMA FOR THE PLAN"""


class HelperCall(BaseModel):
    agent_name: str
    question: str


class ResearchPlan(BaseModel):
    helper_calls: list[HelperCall]
    notes: str | None = None


""" TOP LEVEL MAIN RESEARCH AGENT (CONDUCTOR):"""
main_research_agent = Agent(
    name="Main Stock Research Agent",
    instructions=(
        "You are the main stock research assistant.\n"
        "Given a stock symbol, decide which helper agents to call and what question to ask each.\n"
        "Return ONLY a JSON object that matches the ResearchPlan schema below, using the exact canonical short keys for agent_name: `current_price`, `sentiment`, `technical`, `synthesizer`.\n\n"
        "ResearchPlan JSON schema (example):\n"
        "{\n"
        '  "helper_calls": [\n'
        '    {"agent_name": "current_price", "question": "Get the current price for SYMBOL. Return as JSON: {\\"price\\": 123.45}"},\n'
        '    {"agent_name": "sentiment", "question": "Summarize sentiment and return JSON: {\\"sentiment\\": "bullish", \\"score\\": 0.6}"}\n'
        "  ],\n"
        '  "notes": null\n'
        "}\n\n"
        'Do not wrap the JSON in markdown or explanatory text. If you cannot determine helper calls, return {"helper_calls": [], "notes": "explain here"}.'
    ),
    model="gpt-4o-mini",
    handoffs=[
        current_price_agent,
        sentiment_analysis_agent,
        technical_analysis_agent,
        synthesizer_agent,
    ],
    output_type=ResearchPlan,
)


def normalize_runresult_to_dict(rr: Any) -> dict | None:
    """
    Try to normalize a RunResult (or already-parsed object) into a dict.

    Preference order:
    1. If rr is already a dict, return it.
    2. If rr has attribute `final_output` and it's a dict or JSON string, parse and return.
    3. If rr has `raw_responses` and the first entry is dict or contains JSON, parse and return.
    4. Attempt to regex-extract a JSON object from str(rr) and parse.

    Returns dict on success or None on failure.
    """
    if rr is None:
        return None

    # Already a dict
    if isinstance(rr, dict):
        return rr

    # Pydantic/BaseModel-like
    # 1) Check for a 'final_output' attribute
    rr_dict = getattr(rr, "__dict__", None) or {}
    if (
        isinstance(rr_dict, dict)
        and "final_output" in rr_dict
        and rr_dict["final_output"] is not None
    ):
        fo = rr_dict["final_output"]
        if isinstance(fo, dict):
            return fo
        if isinstance(fo, str):
            try:
                return json.loads(fo)
            except Exception:
                pass

    # 2) Check 'raw_responses'
    if (
        isinstance(rr_dict, dict)
        and "raw_responses" in rr_dict
        and rr_dict["raw_responses"]
    ):
        first = rr_dict["raw_responses"][0]
        if isinstance(first, dict):
            return first
        if isinstance(first, str):
            # Try to find JSON substring
            m = re.search(r"\{.*\}", first, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass

    # 3) last-resort: try parsing the whole string
    try:
        s = str(rr)
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass

    return None


"""DEFINE A RETRY/BACKOFF POLICY TO RUN THE AGENTS"""


async def safe_run(agent, prompt, retries=2):
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


""" DEFINE THE MAIN CLASS CONTAINING THE ORCHESTRATION"""


async def main(stock_symbol: str):
    """
    Orchestrates the stock research flow:

      1) Ask the main_research_agent for a ResearchPlan.
      2) Execute the helper agents specified in that plan (except the synthesizer) concurrently.
      3) Feed the structured results into the synthesizer (if requested in the plan).
    """

    # ------------------------------------------------------------------
    # 1) Ask the main research agent for a plan
    # ------------------------------------------------------------------
    task_prompt = (
        f"Research the stock '{stock_symbol}' around today's close. "
        "Decide which helper agents should be invoked and what question to ask each. "
        "Return a ResearchPlan object."
    )

    # Ask the main research agent and extract a structured ResearchPlan below

    # Map plan.agent_name → actual agent instance
    # The LLM may return full agent names, so we map by both short keys and full names
    agent_map: dict[str, Agent] = {
        # Short keys (for backwards compatibility)
        "current_price": current_price_agent,
        "sentiment": sentiment_analysis_agent,
        "technical": technical_analysis_agent,
        "synthesizer": synthesizer_agent,
        # Full agent names (what the LLM actually returns)
        "Stock Market Current Price Agent": current_price_agent,
        "Stock Sentiment Analysis Agent": sentiment_analysis_agent,
        "Technical Analysis Agent": technical_analysis_agent,
        "Synthesizer Agent": synthesizer_agent,
    }

    # ------------------------------------------------------------------
    plan_raw = await safe_run(main_research_agent, task_prompt)
    print("=== Main agent plan ===")
    print(f"Type of plan_raw: {type(plan_raw)}")

    # Normalize/validate the plan. Prefer strongly-typed ResearchPlan, otherwise
    # try to normalize a RunResult-like object into a dict and validate it.
    if isinstance(plan_raw, ResearchPlan):
        plan = plan_raw
    else:
        # Try to extract dict using helper
        plan_dict = normalize_runresult_to_dict(plan_raw)
        if plan_dict is None:
            rr_dict = getattr(plan_raw, "__dict__", {})
            print(f"RunResult.__dict__ keys: {list(rr_dict.keys())}")
            raise RuntimeError(
                "Failed to extract ResearchPlan from RunResult; ensure the main agent returns the exact JSON ResearchPlan as instructed."
            )

        # Validate/construct ResearchPlan and provide a helpful error if validation fails
        try:
            plan = ResearchPlan(**plan_dict)
        except Exception as e:
            raise RuntimeError(
                f"Parsed ResearchPlan JSON did not validate against schema: {e}\nParsed content: {plan_dict}"
            )

    print(f"Final plan type: {type(plan)}")
    print("Parsed plan:", plan)
    # 2) Run the non-synthesizer helper agents concurrently
    # ------------------------------------------------------------------
    helper_tasks = []  # list of coroutines
    helper_labels: list[str] = (
        []
    )  # parallel list to track which canonical key each task belongs to

    # helper name normalization: map possible LLM-returned names to short canonical keys
    # Normalization mapping for agent names returned by the LLM

    name_to_key = {
        "current_price": "current_price",
        "Stock Market Current Price Agent": "current_price",
        "sentiment": "sentiment",
        "Stock Sentiment Analysis Agent": "sentiment",
        "technical": "technical",
        "Technical Analysis Agent": "technical",
        "synthesizer": "synthesizer",
        "Synthesizer Agent": "synthesizer",
    }

    for call in plan.helper_calls:
        # Normalize the agent name to a canonical key
        canonical_key = name_to_key.get(call.agent_name, call.agent_name)

        # We'll run synthesizer separately, after we have other results
        if canonical_key == "synthesizer":
            continue

        # If the plan requests current_price, prefer calling the deterministic
        # Yahoo price tool rather than invoking the LLM helper. This ensures we
        # get a structured numeric value for synthesis.
        if canonical_key == "current_price":
            helper_labels.append(canonical_key)
            # price_tool.run accepts a text input; we pass the stock symbol so
            # the tool queries Yahoo directly. This returns a dict or None.
            helper_tasks.append(price_tool.run(stock_symbol))
            continue

        agent = agent_map.get(call.agent_name) or agent_map.get(canonical_key)
        if not agent:
            print(f"[Warning] Unknown agent in plan: {call.agent_name}, skipping.")
            continue

        # Prepare the coroutine (not awaited yet)
        helper_labels.append(canonical_key)
        helper_tasks.append(safe_run(agent, call.question))

    results_map: dict[str, object] = {}

    try:
        if helper_tasks:
            # Debug: show what helper tasks we are about to run
            print(f"[debug] helper_labels: {helper_labels}")
            print(f"[debug] number of helper coroutines: {len(helper_tasks)}")

            # Run all helper agents concurrently
            helper_results = await asyncio.gather(*helper_tasks)
            print(f"[debug] helper_results (raw): {helper_results}")

            # Map results back to a dict keyed by canonical label
            for label, result in zip(helper_labels, helper_results):
                print(f"[debug] mapping label={label} -> result={result}")
                results_map[label] = result

            # Extra debug: inspect current_price result if present
            cp = results_map.get("current_price")
            print(f"[debug] results_map['current_price'] = {cp} (type={type(cp)})")
            if isinstance(cp, dict):
                print(f"[debug] current_price keys: {list(cp.keys())}")

        print("=== Helper agent results ===")
        # Print available helper results
        current_val = results_map.get("current_price") or results_map.get(
            "weekly_price"
        )
        sentiment_val = results_map.get("sentiment")
        technical_val = results_map.get("technical")

        if current_val is not None:
            print("Current price result:\n", current_val)
        if sentiment_val is not None:
            print("Sentiment result:\n", sentiment_val)
        if technical_val is not None:
            print("Technical result:\n", technical_val)

        # ------------------------------------------------------------------
        # 3) Combine structured findings and send to the synthesizer
        # ------------------------------------------------------------------
        synthesizer_result = None

        # Only run synthesizer if the plan actually included it
        for call in plan.helper_calls:
            canonical = name_to_key.get(call.agent_name, call.agent_name)
            if canonical != "synthesizer":
                continue

            # Build a structured context object for the synthesizer
            # Note: result objects are RunResult, convert to string representation
            # If current_val is the dict returned by the price tool, extract numeric price
            if isinstance(current_val, dict):
                current_price_val = current_val.get("price")
            else:
                current_price_val = None
                if current_val is not None:
                    try:
                        # Try to coerce a string-like RunResult to a float if possible
                        current_price_val = float(str(current_val))
                    except Exception:
                        current_price_val = str(current_val)

            combined_context = {
                "symbol": stock_symbol,
                "current_price": current_price_val,
                "sentiment": str(sentiment_val) if sentiment_val is not None else None,
                "technical": str(technical_val) if technical_val is not None else None,
            }

            synthesizer_prompt = (
                "You are given structured research findings for a stock in JSON format. "
                "Use them to produce a RecommendationOutput object.\n\n"
                "Research findings:\n"
                f"{combined_context}\n\n"
                "Ensure you follow all constraints defined in RecommendationOutput.\n"
            )

            # If you want, you can also incorporate call.question into the prompt,
            # e.g. f\"{call.question}\n\n{combined_context}\"
            synthesizer_result = await safe_run(synthesizer_agent, synthesizer_prompt)
            break  # Only call synthesizer once

        if synthesizer_result is not None:
            print("=== Synthesizer recommendation ===")
            # synthesizer_result is a RunResult, convert to string
            print(str(synthesizer_result))
        else:
            print("[Info] No synthesizer step specified in the plan.")

    except Exception as e:
        print("Error running sub-agents or synthesizer:", e)


""" INVOKE MAIN"""
if __name__ == "__main__":
    stock = input("Enter stock ticker (e.g., TSLA): ").strip().upper()
    if not stock:
        print("No stock provided, exiting.")
    else:
        asyncio.run(main(stock))
