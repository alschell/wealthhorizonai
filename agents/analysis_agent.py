import matplotlib.pyplot as plt
import io
import base64
from functools import lru_cache
import logging
import pandas as pd
import numpy as np
import random
from ..core.portfolio import Portfolio
from ..utils.helpers import calculate_esg_score

logger = logging.getLogger(__name__)

class AnalysisAgent:
    def __init__(self, state):
        self.state = state

    @lru_cache(maxsize=32)
    async def get_performance(self, params=None):
        market_returns = self.state.market_data.mean(axis=1).pct_change().dropna()
        perf = {}
        for name, p in self.state.portfolios.items():
            r = p.returns()
            income = p.simulate_income()
            attr = p.attribution()
            perf[name] = {
                'annual_return': r.mean() * 252,
                'volatility': r.std() * np.sqrt(252),
                'sharpe': (r.mean() / r.std()) * np.sqrt(252) if r.std() != 0 else 0,
                'beta': p.beta(),
                'income': income,
                'drivers': {'market': r.mean() * 0.7, 'currency': random.uniform(-0.01, 0.01), 'income': income / p.value(self.state.market_data.index[-1])},
                'attribution': attr,
                'esg_score': calculate_esg_score(p)
            }
        return perf

    async def compare_portfolios(self, params=None):
        df = pd.DataFrame(await self.get_performance(params)).T
        if len(df) > 1:
            diff = df.diff().iloc[1]
            print(f"Performance Differences (P2 vs P1): Return {diff['annual_return']:.2%}, Volatility {diff['volatility']:.2%}")
        print("Portfolio Comparison:\n", df)
        return df.to_string()

    async def generate_report(self, params=None):
        report = "Wealth Horizon Comprehensive Report\n"
        report += f"Date: {datetime.now()}\n"
        report += "Performance Metrics:\n" + await self.compare_portfolios(params) + "\n"
        report += "Asset Allocation:\n" + str(await self.get_asset_allocation(params)) + "\n"
        report += "Macro Scenarios:\n" + str(self.state.scenarios) + "\n"
        report += "Concentration Risk:\n" + str(await self.delegate('risk_scenario', 'get_concentration_risk')) + "\n"
        with open('comprehensive_report.txt', 'w') as f:
            f.write(report)
        return "Comprehensive report exported to comprehensive_report.txt"

    async def get_holdings_cash(self, params=None):
        holdings_cash = {}
        for name, p in self.state.portfolios.items():
            holdings_cash[name] = {'holdings': p.holdings, 'cash': p.cash}
        return holdings_cash

    async def get_asset_allocation(self, params=None):
        allocations = {}
        for name, p in self.state.portfolios.items():
            alloc = {}
            for asset, info in p.holdings.items():
                class_ = self.state.asset_classes.get(asset, 'Other')
                alloc[class_] = alloc.get(class_, 0) + (info['qty'] * self.state.market_data[asset].iloc[-1] / p.value(self.state.market_data.index[-1]))
            allocations[name] = alloc
        return allocations

    async def optimize_portfolios(self, params=None):
        for p in self.state.portfolios.values():
            p.optimize_allocation()
        return "Portfolios optimized with asset class constraints."

    async def create_graphic(self, params):
        items = params.get('items', [])
        fig, ax = plt.subplots(figsize=(12, 8))
        for item_type, name in items:
            label = f"{item_type.capitalize()} {name}"
            if item_type == 'portfolio':
                p = self.state.portfolios.get(name)
                if p:
                    series = p.value_series() / p.value_series().iloc[0] * 100 if p.value_series().iloc[0] != 0 else pd.Series(100, index=self.state.market_data.index)
                    ax.plot(series, label=label)
            elif item_type in ['stock', 'index']:
                if name in self.state.market_data:
                    series = self.state.market_data[name] / self.state.market_data[name].iloc[0] * 100
                    ax.plot(series, label=label)
        ax.legend()
        ax.set_title("Performance Overlay Comparison (Normalized to 100)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Performance (%)")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return f"data:image/png;base64,{img_base64}"
