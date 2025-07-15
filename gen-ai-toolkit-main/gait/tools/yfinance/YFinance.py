from pydantic import BaseModel, Field


class YFinance(BaseModel):
    """
    Use this function to get the current stock price for a given symbol.
    """

    ticker: str = Field(..., description="The stock ticker symbol.")

    def __call__(self, *args, **kwargs) -> str:
        try:
            import yfinance as yf

            stock = yf.Ticker(self.ticker)
            # Use "regularMarketPrice" for regular market hours, or "currentPrice" for pre/post market
            current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
            return f"{current_price:.4f}" if current_price else f"Could not fetch current price for {self.ticker}"
        except Exception as e:
            return f"Error fetching current price for {self.ticker}: {e}"
