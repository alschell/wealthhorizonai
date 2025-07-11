import torch
import torch.nn.functional as F
import logging
from ..utils.ml_models import predictor, optimizer, save_model

logger = logging.getLogger(__name__)

class ForecastingAgent:
    def __init__(self, state):
        self.state = state

    async def forecast_returns(self, params=None):
        forecasts = {}
        for asset in self.state.asset_classes.keys():
            past_data = self.state.market_data[asset].pct_change().dropna()[-10:].values
            if len(past_data) < 10:
                continue
            input_tensor = torch.tensor(past_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = predictor(input_tensor).item()
            forecasts[asset] = pred
        dummy_input = torch.randn(64, 10)
        dummy_target = torch.randn(64, 1)
        pred_output = predictor(dummy_input)
        loss = F.mse_loss(pred_output, dummy_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_model(predictor)
        return forecasts
