# Wealth Horizon Multi-Agent AI

## Setup
- `pip install -r requirements.txt`
- Set env vars: `export API_KEY=your_key; export AZURE_STORAGE_CONNECTION_STRING=your_conn`
- Run: `python api.py`

## Deployment
- Build Docker: `docker build -t wealth-horizon .`
- Push to Azure Container Registry, deploy to App Service.
- Use Azure ML for model training/scaling.
