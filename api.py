from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.security import APIKeyHeader
import uvicorn
import os
from dotenv import load_dotenv
from core.coordinator import CoordinatorAgent
from core.state import SharedState
from core.portfolio import Portfolio

load_dotenv()

app = FastAPI(title="Wealth Horizon AI")

state = SharedState()
coord = CoordinatorAgent(state)

# Sample portfolios
p1 = Portfolio("EquityFocused", "Growth", {
    'AAPL': {'qty': 100, 'asset_class': 'Equity', 'region': 'US', 'currency': 'USD'},
    'BOND_US': {'qty': 200, 'asset_class': 'Fixed Income', 'region': 'US', 'currency': 'USD'},
    'GOLD': {'qty': 50, 'asset_class': 'Commodities', 'region': 'Global', 'currency': 'USD'},
    'PE_FUND': {'qty': 30, 'asset_class': 'Private Equity', 'region': 'US', 'currency': 'USD'},
    'REAL_ESTATE': {'qty': 20, 'asset_class': 'Real Estate', 'region': 'Europe', 'currency': 'EUR'},
    'BTC': {'qty': 1, 'asset_class': 'Cryptocurrency', 'region': 'Global', 'currency': 'BTC'},
    'ART_COLLECTION': {'qty': 5, 'asset_class': 'Passion Assets', 'region': 'Global', 'currency': 'USD'}
}, {'USD': 10000, 'EUR': 5000})
p2 = Portfolio("BalancedAlt", "Diversified", {
    'HEDGE_FUND': {'qty': 40, 'asset_class': 'Hedge Fund', 'region': 'US', 'currency': 'USD'},
    'OIL': {'qty': 60, 'asset_class': 'Commodities', 'region': 'Global', 'currency': 'USD'},
    'ETH': {'qty': 10, 'asset_class': 'Cryptocurrency', 'region': 'Global', 'currency': 'USD'},
    'WINE_VINTAGE': {'qty': 15, 'asset_class': 'Passion Assets', 'region': 'Europe', 'currency': 'EUR'},
    'BOND_CORP': {'qty': 150, 'asset_class': 'Fixed Income', 'region': 'US', 'currency': 'USD'}
}, {'GBP': 8000, 'CHF': 3000})
state.portfolios = {'P1': p1, 'P2': p2}
state.hierarchy['holistic']['groups']['Family_Smith'] = {'individuals': {'Member_John': {'portfolios': [p1]}, 'Member_Jane': {'portfolios': [p2]}}}
state.hierarchy['holistic']['groups']['Client_ABC_WealthMgr'] = {'individuals': {'Client_XYZ': {'portfolios': [p1, p2]}}}

API_KEY = os.getenv('API_KEY')

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.get("/query")
async def query_endpoint(text: str = Query(...), api_key: str = Depends(get_api_key)):
    result = await coord.process_query(text)
    return {"result": result}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))
