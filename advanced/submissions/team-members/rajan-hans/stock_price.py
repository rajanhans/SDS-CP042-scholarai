# This code fetches the stock price for a given symbol from Tavily's API.

from urllib import response
import requests
import os
from dotenv import load_dotenv


print("Program started")
load_dotenv()
TAVILY_KEY = os.getenv("TAVILY_KEY")


def get_stock_price(symbol, TAVILY_KEY):
    """
    Fetches the current stock price for a given symbol from Tavily.

    Parameters:
    symbol (str): Stock ticker symbol, e.g., "AAPL"
    api_key (str): Your Tavily API key

    Returns:
    float: Current stock price, or None if the request fails
    """

    url = f"https://api.tavily.com/v1/stocks/{symbol}/quote"  # Example endpoint
    headers = {"Authorization": f"Bearer {TAVILY_KEY}"}

    from tavily import TavilyClient

    client = TavilyClient(TAVILY_KEY)
    response = client.search(
        query=f"what is the stock price for {symbol}", include_answer="advanced"
    )
    print(response)
    return response


if __name__ == "__main__":
    symbol = input("Enter stock ticker (e.g., TSLA): ").strip().upper()
    if not symbol:
        print("No stock provided, exiting.")
    else:
        price = get_stock_price(symbol, TAVILY_KEY)
        if price is not None:
            print(f"The current price of {symbol} is ${price}")
        else:
            print(f"Failed to fetch the price for {symbol}.")
