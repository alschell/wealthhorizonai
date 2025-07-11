import random
from datetime import datetime
from agents.analysis_agent import AnalysisAgent
from agents.compliance_agent import ComplianceAgent
from agents.forecasting_agent import ForecastingAgent
from agents.research_agent import ResearchAgent
from agents.risk_agent import RiskAgent
from agents.trade_agent import TradeAgent
import logging

logger = logging.getLogger(__name__)

class CoordinatorAgent:
    def __init__(self, state):
        self.state = state
        self.agents = {
            'analysis': AnalysisAgent,
            'compliance': ComplianceAgent,
            'forecasting': ForecastingAgent,
            'research': ResearchAgent,
            'risk_scenario': RiskAgent,
            'trade': TradeAgent
        }

    async def process_query(self, query):
        logger.info(f"Processing query: {query}")
        query_lower = query.lower()
        if 'performance' in query_lower:
            perf = await self.delegate('analysis', 'get_performance')
            critique = await self.delegate('compliance', 'check_compliance', {'perf': perf})
            return {"performance": perf, "critique": critique}
        elif 'scenario' in query_lower:
            drop = -0.05 if '5%' in query_lower else -0.10
            return await self.delegate('risk_scenario', 'analyze_scenario', {'drop': drop})
        elif 'trade ideas' in query_lower:
            return await self.delegate('trade', 'generate_ideas')
        elif 'sell' in query_lower or 'buy' in query_lower:
            return await self.delegate('trade', 'execute_trade', {'action': query})
        elif 'report' in query_lower:
            return await self.delegate('analysis', 'generate_report')
        elif 'compare' in query_lower and not 'graphic' in query_lower:
            return await self.delegate('analysis', 'compare_portfolios')
        elif 'autopilot' in query_lower:
            return await self.delegate('trade', 'autopilot_rebalance')
        elif 'forecast' in query_lower:
            return await self.delegate('forecasting', 'forecast_returns')
        elif 'compliance' in query_lower:
            return await self.delegate('compliance', 'check_compliance')
        elif 'optimize' in query_lower:
            return await self.delegate('analysis', 'optimize_portfolios')
        elif 'holdings' in query_lower or 'cash' in query_lower:
            return await self.delegate('analysis', 'get_holdings_cash')
        elif 'asset allocation' in query_lower:
            return await self.delegate('analysis', 'get_asset_allocation')
        elif 'concentration' in query_lower or 'volatility' in query_lower:
            return await self.delegate('risk_scenario', 'get_concentration_risk')
        elif 'macro scenarios' in query_lower:
            return await self.delegate('risk_scenario', 'get_macro_scenarios')
        elif 'graphic' in query_lower or 'compare' in query_lower and ('portfolio' in query_lower or 'stock' in query_lower or 'index' in query_lower):
            items = []
            if 'portfolio abc' in query_lower or 'p1' in query_lower:
                items.append(('portfolio', 'P1'))
            if 'portfolio xyz' in query_lower or 'p2' in query_lower:
                items.append(('portfolio', 'P2'))
            if 's&p 500' in query_lower or 'sp500' in query_lower:
                items.append(('index', 'SP500'))
            if 'dow' in query_lower or 'djia' in query_lower:
                items.append(('index', 'DJIA'))
            if 'tesla' in query_lower or 'tsla' in query_lower:
                items.append(('stock', 'TSLA'))
            if 'apple' in query_lower or 'aapl' in query_lower:
                items.append(('stock', 'AAPL'))
            if not items:
                items = [('portfolio', 'P1'), ('index', 'SP500')]
            return await self.delegate('analysis', 'create_graphic', {'items': items})
        else:
            market_data = await self.delegate('research', 'get_market_data')
            forecasts = await self.delegate('forecasting', 'forecast_returns', {'market_data': market_data})
            perf = await self.delegate('analysis', 'get_performance', {'forecasts': forecasts})
            risk = await self.delegate('risk_scenario', 'analyze_scenario', {'drop': -0.05})
            ideas = await self.delegate('trade', 'generate_ideas', {'risk': risk})
            return f"End-to-end analysis: Performance {perf}, Forecasts {forecasts}, Risk {risk}, Ideas {ideas}"

    async def delegate(self, agent_name, method, params=None):
        agent_class = self.agents[agent_name]
        agent = agent_class(self.state)
        return await getattr(agent, method)(params or {})
