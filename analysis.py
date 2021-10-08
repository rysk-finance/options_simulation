from storage import get_results
import cryptocompare
import pandas as pd
from empyrical import max_drawdown, alpha_beta

def generate_risk_return_metrics_historical():
    underlying = get_daily_returns_underlying()
    results = get_results()
    results = [pd.DataFrame(x) for x in results]
    for i, result in enumerate(results):
        result['time'] = pd.to_datetime(result['timestamp'])
        result = result.set_index('time')
        result = result.resample('d').mean()
        merged = result.join(underlying[['date', 'daily_change', 'close']].dropna())
        merged.rename(columns={'daily_change':'daily_return_underlying', 'open': 'open_underlying'}, inplace=True)
        merged['daily_return'] = merged['equity'].pct_change()
        merged = merged.dropna()
        # compare underlying and portfolio
        daily_volatility = merged['daily_return'].std()
        daily_volatility_underlying = merged['daily_return_underlying'].std()
        avg_daily_return = merged['daily_return'].mean()
        avg_daily_return_underlying = merged['daily_return_underlying'].mean()
        sharpe = avg_daily_return / daily_volatility * (356**0.5)
        sharpe_underlying = avg_daily_return_underlying / daily_volatility_underlying * (356**0.5)
        max_dd = max_drawdown(merged['daily_return'])
        max_dd_underlying = max_drawdown(merged['daily_return_underlying'])
        alpha, beta = alpha_beta(merged['daily_return'], merged['daily_return_underlying'])
        # calc total return
        total_return = (merged.iloc[-1]['equity'] - merged.iloc[0]['equity']) / merged.iloc[0]['equity']
        total_return_underlying = (merged.iloc[-1]['open_underlying'] - merged.iloc[0]['open_underlying']) / merged.iloc[0]['open_underlying']
        results[i] = {
            'data': merged,
            'alpha': alpha,
            'beta': beta,
            'max_drawdown': max_dd,
            'max_drawdown_underlying': max_dd_underlying,
            'sharpe': sharpe,
            'sharpe_underlying': sharpe_underlying,
            'avg_daily_return': avg_daily_return,
            'avg_daily_return_underlying': avg_daily_return_underlying,
            'daily_volatility': daily_volatility,
            'daily_volatility_underlying': daily_volatility_underlying,
            'total_return': total_return,
            'total_return_underlying': total_return_underlying
        }
    return results


def get_daily_returns_underlying():
    eth_daily = cryptocompare.get_historical_price_day('ETH')
    eth_df = pd.DataFrame(eth_daily)
    eth_df['daily_change'] = eth_df.close.pct_change(1)
    eth_data = eth_df['daily_change'].dropna()
    eth_df['time'] = (eth_df['time']).apply(lambda x: pd.datetime.fromtimestamp(x).date())
    eth_df['time'] = pd.to_datetime(eth_df['time'])
    eth_df['date'] = pd.to_datetime(eth_df['time']).dt.date
    eth_df.set_index('time', inplace=True)
    return eth_df
