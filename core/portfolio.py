import pandas as pd
import numpy as np
import random
import logging
from ..utils.ml_models import state_dim, action_dim
from ..utils.ml_models import prices, dates, exchange_rates

logger = logging.getLogger(__name__)

class Portfolio:
    def __init__(self, name, strategy, holdings, cash, target_allocation=None):
        self.name = name
        self.strategy = strategy
        self.holdings = holdings
        self.cash = cash
        self.transactions = []
        self.target_allocation = target_allocation or {asset: 1/len(assets) for asset in assets}

    def value(self, date):
        val = sum(self.cash.get(c, 0) * exchange_rates.get(c, 1.0) for c in self.cash)
        for asset, info in self.holdings.items():
            val += info['qty'] * prices[asset].loc[date] * exchange_rates.get(info['currency'], 1.0)
        return val

    def returns(self):
        values = pd.Series([self.value(d) for d in dates if d in prices.index], index=[d for d in dates if d in prices.index])
        return values.pct_change().dropna()

    def value_series(self):
        return pd.Series([self.value(d) for d in dates if d in prices.index], index=[d for d in dates if d in prices.index])

    def beta(self):
        market_returns = prices.mean(axis=1).pct_change().dropna()
        port_returns = self.returns()
        cov = port_returns.cov(market_returns)
        var = market_returns.var()
        return cov / var if var != 0 else 0

    def simulate_income(self):
        income = 0
        for asset, info in self.holdings.items():
            if info['asset_class'] in ['Equity', 'Fixed Income']:
                income += info['qty'] * random.uniform(0.01, 0.05) * prices[asset].iloc[-1]
            elif info['asset_class'] == 'Real Estate':
                income += info['qty'] * random.uniform(0.04, 0.08) * prices[asset].iloc[-1]
        return income

    def attribution(self):
        benchmark_returns = prices['SP500'].pct_change().dropna().mean()
        port_returns = self.returns().mean()
        allocation_effect = random.uniform(0, 0.02)
        selection_effect = port_returns - benchmark_returns
        return {'allocation': allocation_effect, 'selection': selection_effect, 'total': allocation_effect + selection_effect}

    def optimize_allocation(self):
        asset_returns = prices.pct_change().dropna().mean() * 252
        cov = prices.pct_change().dropna().cov() * 252
        weights = np.random.dirichlet(np.ones(len(assets)))
        crypto_idx = [assets.index(a) for a in ['BTC', 'ETH'] if a in assets]
        weights[crypto_idx] = np.minimum(weights[crypto_idx], 0.2)
        weights /= weights.sum()
        self.target_allocation = dict(zip(assets, weights))
