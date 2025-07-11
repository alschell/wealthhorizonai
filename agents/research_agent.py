import logging

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, state):
        self.state = state

    async def get_market_data(self, params=None):
        return self.state.market_data.tail(20).to_dict()
