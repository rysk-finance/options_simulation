from storage import get_results
import cryptocompare
import pandas as pd

def generate_risk_return_metrics_historical():
    results = get_results()

def get_daily_returns_underlying():
    eth_daily = cryptocompare.get_historical_price_day('ETH')
    eth_df = pd.DataFrame(eth_daily)
    eth_df['daily_change'] = eth_df.close.pct_change(1)
    eth_data = eth_df['daily_change'].dropna()
    eth_df['date'] = (eth_df['time']).apply(lambda x: pd.datetime.fromtimestamp(x).date())
    eth_df['date'] = pd.to_datetime(eth_df['date'])
    eth_df.set_index('date', inplace=True)
    return eth_df
