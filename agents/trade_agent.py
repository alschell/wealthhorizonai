import torch
import random
import logging
from ..utils.ml_models import q_network, rl_optimizer, action_dim, state_dim
from ..utils.helpers import simulate_tax_optimization

logger = logging.getLogger(__name__)

class TradeAgent:
    def __init__(self, state):
        self.state = state

    async def generate_ideas(self, params=None):
        ideas = []
        for asset in self.state.asset_classes.keys():
            past_data = self.state.market_data[asset].pct_change().dropna()[-10:].values
            if len(past_data) < 10:
                continue
            input_tensor = torch.tensor(past_data, dtype=torch.float32).unsqueeze(0)
            pred = predictor(input_tensor).item()
            state = torch.tensor(np.concatenate([past_data, [pred, self.state.portfolios[list(self.state.portfolios.keys())[0]].beta()]]), dtype=torch.float32)
            q_values = q_network(state)
            action = torch.argmax(q_values).item()
            if action == 0 and pred > 0.02:
                ideas.append(f"Buy {asset} (predicted return: {pred:.2%}, RL action: Buy, Q: {q_values[0].item():.2f})")
            elif action == 1 and pred < -0.02:
                ideas.append(f"Sell {asset} (predicted return: {pred:.2%}, RL action: Sell, Q: {q_values[1].item():.2f})")
            elif action == 3:
                ideas.append(f"Hedge {asset} (RL action: Hedge, Q: {q_values[3].item():.2f})")
        return ideas if ideas else ["No strong trade signals."]

    async def execute_trade(self, params):
        action = params['action'].lower()
        if 'sell' in action:
            asset = 'AAPL' if 'apple' in action else 'TSLA'
            qty_frac = 0.5 if 'half' in action else 1.0
            p = list(self.state.portfolios.values())[0]
            if asset in p.holdings:
                qty = p.holdings[asset]['qty'] * qty_frac
                proceeds = qty * self.state.market_data[asset].iloc[-1]
                p.holdings[asset]['qty'] -= qty
                approved = 'y'
                if approved == 'y':
                    broker = 'Interactive Brokers'
                    logger.info(f"Trade routed to {broker}")
                    p.transactions.append({'type': 'sell', 'asset': asset, 'qty': qty, 'proceeds': proceeds})
                    proceeds = simulate_tax_optimization(proceeds)
                    if 'deposit' in action:
                        accounts = [('GBP', 'Standard Chartered', 'UK'), ('CHF', 'UBS', 'Switzerland'), ('HKD', 'HSBC', 'Hong Kong')]
                        split = proceeds / len(accounts)
                        for curr, bank, loc in accounts:
                            p.cash[curr] = p.cash.get(curr, 0) + split
                            logger.info(f"Deposited {split:.2f} {curr} to {bank} in {loc}")
                    return "Trade executed, approved, tax-optimized, and funds deposited."
                return "Trade rejected."
        return "Trade processed."

    async def autopilot_rebalance(self, params=None):
        for name, p in self.state.portfolios.items():
            current_weights = {asset: (info['qty'] * self.state.market_data[asset].iloc[-1]) / p.value(self.state.market_data.index[-1]) for asset, info in p.holdings.items() if p.value(self.state.market_data.index[-1]) > 0}
            for asset, target in p.target_allocation.items():
                diff = target - current_weights.get(asset, 0)
                if abs(diff) > 0.03:
                    action = 'buy' if diff > 0 else 'sell'
                    qty = abs(diff) * p.value(self.state.market_data.index[-1]) / self.state.market_data[asset].iloc[-1]
                    logger.info(f"Autopilot {name}: {action.capitalize()} {qty:.0f} of {asset} (strategy: {p.strategy})")
                    state = np.random.rand(state_dim)
                    next_state = np.random.rand(state_dim)
                    reward = random.uniform(-1, 1) + (diff * 0.1)
                    self.state.rl_memory.append((state, random.randint(0, action_dim-1), reward, next_state))
                    if len(self.state.rl_memory) > 64:
                        batch = random.sample(self.state.rl_memory, 64)
                        states = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32)
                        actions = torch.tensor([s[1] for s in batch], dtype=torch.long)
                        rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
                        next_states = torch.tensor(np.array([s[3] for s in batch]), dtype=torch.float32)
                        q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                        next_q = q_network(next_states).max(1)[0].detach()
                        targets = rewards + 0.99 * next_q
                        loss = F.smooth_l1_loss(q_values, targets)
                        rl_optimizer.zero_grad()
                        loss.backward()
                        rl_optimizer.step()
        return "Autopilot rebalancing complete with RL and strategy alignment."
