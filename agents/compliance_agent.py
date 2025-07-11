import random
import logging

logger = logging.getLogger(__name__)

class ComplianceAgent:
    def __init__(self, state):
        self.state = state

    async def check_compliance(self, params=None):
        risks = await self.state.delegate('risk_scenario', 'get_concentration_risk')
        alloc = await self.state.delegate('analysis', 'get_asset_allocation')
        checks = []
        for name in self.state.portfolios:
            if risks.get(name, {})['hhi'] > 0.20:
                checks.append(f"Warning: {name} high concentration (HHI > 0.20)")
            if alloc.get(name, {}).get('Cryptocurrency', 0) > 0.15:
                checks.append(f"Warning: {name} exceeds crypto limit (15%)")
            if any('crypto' in alloc.get(name, {}) for alloc in alloc.values()):
                checks.append(f"Note: Crypto holdings flagged for AML review")
        return "Compliance check: " + "; ".join(checks) if checks else "All clear."
