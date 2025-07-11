import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class RiskAgent:
    def __init__(self, state):
        self.state = state

    async def analyze_scenario(self, params):
        drop = params['drop']
        impacts = {}
        for name, p in self.state.portfolios.items():
            sim_returns = np.random.normal(p.returns().mean(), p.returns().std(), 5000) + drop * p.beta()
            esg_score = random.uniform(0.5, 0.9) if 'climate' in params.get('name', '').lower() else 1.0
            impacts[name] = {
                'mean_impact': np.mean(sim_returns) * esg_score,
                'var_95': np.percentile(sim_returns, 5),
                'hedge_suggestion': random.choice([s['hedge'] for s in self.state.scenarios]),
                'tax_optimization': 'Consider tax-loss harvesting if impact negative'
            }
        return impacts

    async def get_concentration_risk(self, params=None):
        risks = {}
        for name, p in self.state.portfolios.items():
            values = np.array([info['qty'] * self.state.market_data[asset].iloc[-1] for asset, info in p.holdings.items() if 'qty' in info])
            weights = values / values.sum() if values.sum() != 0 else np.zeros_like(values)
            hhi = np.sum(weights ** 2)
            volatility = p.returns().std() * np.sqrt(252)
            risks[name] = {'hhi': hhi, 'diversification_score': 1 - hhi, 'annual_volatility': volatility}
        print("Concentration & Volatility Risks:", risks)
        return risks

    async def get_macro_scenarios(self, params=None):
        sorted_scenarios = sorted(self.state.scenarios, key=lambda x: x['prob'], reverse=True)
        return sorted_scenarios
