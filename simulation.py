import pandas as pd
import numpy as np
import time
import datetime
from glob import glob
import math
import matplotlib.pyplot as plt
import wdb

pd.options.mode.chained_assignment = None

COLUMNS = ["symbol", "timestamp", "type", "strike_price", "expiration", "underlying_price", "bid_price", "bid_amount",
           "ask_price", "ask_amount", "mark_price", "delta", "theta"]


def ensure_option_series(option):
    return option if (isinstance(option, pd.Series)) else option.iloc[0]


def negative_deltas(df):
    return df[df.delta < 0]


def positive_deltas(df):
    return df[df.delta > 0]


def filter_deltas(df):
    return df[((df.delta > -0.50) & (df.delta < -0.25)) | ((df.delta < 0.5) & (df.delta > 0.25))]


def filter_expiration(df):
    return df[(df['days_to_expiration'] > datetime.timedelta(days=45)) & (
                df['days_to_expiration'] < datetime.timedelta(days=70))]


def filter_datetime(df, start, end):
    return df[(df['datetime'] >= start) & (df['datetime'] <= end)]


class Simulation:
    def __init__(self, starting_capital=1000000, max_epoch_allocation=0.10):
        self.cash = starting_capital
        self.assets = 0
        self.liabilities = 0
        self.collateral_locked = 0
        self.equity = starting_capital
        self.max_epoch_allocation = max_epoch_allocation
        self.portfolio_delta = 0
        self.positions = pd.DataFrame(
            columns=COLUMNS + ['num_contracts', 'position_delta', 'collateral_locked', 'liability_amount'])
        self.files = self.get_files()
        self.current_time = None
        self.end_sample_time = None
        self.statistics_overtime = []

    def run(self):
        failed = None
        for file in self.files:
            try:
                self.load_file_to_dataframe(file)
            except:
                print(f'loading {file} failed')
                failed = True
                continue
            if failed:
                self.set_times(override=True)
                failed = None
            else:
                self.set_times()
            filtered = self.get_filtered_options()
            while not filtered.empty:
                self.add_positions(filtered)
                self.mark_portfolio()
                self.find_and_close_positions()
                self.mark_portfolio()
                self.timestamp_statistics()
                self.set_times()
                print(f'current time: {self.current_time}')
                print(f'portfolio delta: {self.portfolio_delta}')
                print(f'cash: {self.cash}, equity: {self.equity}, liabilities: ${self.liabilities}')
                filtered = self.get_filtered_options()
        self.plot()

    def allocate_funds(self, option, cash):
        option_series = option if (isinstance(option, pd.Series)) else option.iloc[0]
        required_collateral = self.get_collateral_required(option_series)
        contracts = math.floor(cash / required_collateral)
        # make sure not trying to sell more than possible at price
        contracts = contracts if (contracts <= option_series['bid_amount']) else math.floor(option_series['bid_amount'])
        self.write_allocation(option_series, contracts)

    def add_positions(self, filtered_book):
        # add some variance by drawing at random from filtered down set
        positive = positive_deltas(filtered_book)
        negative = negative_deltas(filtered_book)
        deployable_cash = self.cash * self.max_epoch_allocation
        if len(self.positions) == 0:
            sample = positive.sample()
            self.allocate_funds(sample, deployable_cash * 0.5)
            self.update_state()
            sample = negative.sample()
            sample = sample.iloc[0]
            self.delta_hedge_from_series(sample, deployable_cash * 0.5)

        if self.portfolio_delta >= 0:
            # deltas will be inverted due to writing
            sample = positive.sample()
            self.delta_hedge_from_series(sample, deployable_cash)
        else:
            sample = negative.sample()
            self.delta_hedge_from_series(sample, deployable_cash)

    def close_positions(self, options):
        symbols = set(options['symbol'])
        to_drop = []
        for i, row in self.positions.iterrows():
            if row['symbol'] in symbols:
                # can make more realistic using ask_size
                ask_price_usd = row['ask_price'] * row['underlying_price']
                cost_to_close = abs(ask_price_usd * row['num_contracts'])
                if cost_to_close <= row['collateral_locked']:
                    self.cash = float(self.cash + row['collateral_locked'] - cost_to_close)
                    to_drop.append(i)
        self.positions.drop(to_drop, inplace=True)
        if len(to_drop) > 0:
            print('\033[96m' + f'closed out {len(to_drop)} positions' + '\033[0m')
        self.update_state()

    def delta_hedge_from_series(self, option, cash):
        option_series = option if (isinstance(option, pd.Series)) else option.iloc[0]
        contracts_needed = abs(self.portfolio_delta / option['delta'])
        required_collateral = self.get_collateral_required(option_series)
        contracts = math.floor(contracts_needed)
        contracts_threshold = math.floor(cash / required_collateral)
        contracts = contracts if contracts <= contracts_threshold else contracts_threshold
        if contracts > 0:
            contracts = contracts if (contracts <= option_series['bid_amount']) else math.floor(
                option_series['bid_amount'])
        else:
            return
        self.write_allocation(option, contracts)

    def load_file_to_dataframe(self, file_path):
        t = time.process_time()
        df = pd.read_csv(file_path, compression='gzip', usecols=COLUMNS)
        eth_only = df[df['symbol'].str.contains("ETH")]
        eth_only['datetime'] = pd.to_datetime(eth_only['timestamp'] * 1000)
        eth_only['expiration_datetime'] = pd.to_datetime(eth_only['expiration'] * 1000)
        eth_only['days_to_expiration'] = eth_only['expiration_datetime'] - eth_only['datetime']
        eth_only.dropna(inplace=True)
        self.current_day_orderbook = eth_only
        elapsed_time = time.process_time() - t
        print(f'{file_path} loaded in {elapsed_time} seconds')

    def find_and_close_positions(self):
        # under 45 days gamma becomes high
        expiring = self.get_short_expiration_positions()
        if self.portfolio_delta >= 0:
            expiring = positive_deltas(expiring)
        else:
            expiring = negative_deltas(expiring)
        if not expiring.empty:
            self.close_positions(expiring)

    def get_collateral_required(self, option_series):
        option_series = ensure_option_series(option_series)
        option_type = option_series['type']
        if option_type == 'call':
            # assume posting underlying value to meet call writing requirements
            return option_series['underlying_price']
        else:
            return option_series['strike_price']

    def get_filtered_options(self, time_only=False):
        # get current slice of datetime in order book
        clone = self.current_day_orderbook.copy()
        clone = filter_datetime(clone, self.current_time, self.end_sample_time)
        if time_only:
            return clone.sort_values(by='timestamp')
        clone = filter_deltas(clone)
        clone = filter_expiration(clone)
        return clone.sort_values(by='timestamp')

    def get_files(self):
        files = glob("datasets/*.csv.gz")
        files.sort()
        return files

    def get_short_expiration_positions(self, days=45):
        return self.positions[self.positions['days_to_expiration'] < datetime.timedelta(days=days)]

    def mark_portfolio(self):
        filtered = self.get_filtered_options(time_only=True)
        for i, row in self.positions.iterrows():
            symbol = row['symbol']
            f_idx = filtered.symbol.eq(symbol).idxmax()
            filtered_row = filtered.loc[f_idx]
            self.positions.at[i, 'delta'] = filtered_row['delta']
            self.positions.at[i, 'mark_price'] = filtered_row['mark_price']
            self.positions.at[i, 'underlying_price'] = filtered_row['underlying_price']
            self.positions.at[i, 'ask_price'] = filtered_row['ask_price']
            self.positions.at[i, 'days_to_expiration'] = filtered_row['expiration_datetime'] - self.current_time
        self.positions['position_delta'] = self.positions['num_contracts'] * self.positions['delta']
        self.positions['liability_amount'] = abs(
            self.positions['num_contracts'] * self.positions['mark_price'] * self.positions['underlying_price'])
        self.update_state()

    def set_times(self, override=False):
        first_time = self.current_day_orderbook.iloc[0]['datetime']
        if self.current_time and not override:
            excess_time_gap = first_time - self.current_time > datetime.timedelta(days=1, hours=12)
            self.current_time = self.current_time + pd.DateOffset(hours=3) if not excess_time_gap else first_time
        else:
            self.current_time = first_time

        self.end_sample_time = self.current_time + pd.DateOffset(minutes=30)

    def timestamp_statistics(self):
        self.statistics_overtime.append({
            'timestamp': self.current_time,
            'equity': self.equity,
            'cash': self.cash,
            'liabilities': self.liabilities,
            'collateral_locked': self.collateral_locked
        })

    def update_equity(self):
        self.equity = self.cash + self.collateral_locked - self.liabilities

    def update_collateral_locked(self):
        self.collateral_locked = self.positions['collateral_locked'].sum()

    def update_liabilities(self):
        self.liabilities = self.positions['liability_amount'].sum()

    def update_portfolio_delta(self):
        self.portfolio_delta = self.positions['position_delta'].sum()

    def update_state(self):
        self.update_portfolio_delta()
        self.update_liabilities()
        self.update_collateral_locked()
        self.update_equity()

    def write_allocation(self, option_series, contracts):
        bid_price_usd = option_series['bid_price'] * option_series['underlying_price']
        allocation_amount = contracts * self.get_collateral_required(option_series)
        premium_received = contracts * bid_price_usd
        # invert signs for writing
        option_series['num_contracts'] = option_series['num_contracts'] + -(
            contracts) if 'num_contracts' in option_series else -(contracts)
        option_series['position_delta'] = -(contracts * option_series['delta'])
        option_series['collateral_locked'] = allocation_amount
        option_series['liability_amount'] = abs(
            contracts * option_series['mark_price'] * option_series['underlying_price'])
        self.positions = self.positions.append(option_series)
        self.cash = float(self.cash - allocation_amount + premium_received)
        self.update_state()

    def plot(self):
        stats = pd.DataFrame(self.statistics_overtime)
        for col in stats.columns[1:]:
            ax = stats.plot(stats.index[0], col)
            plt.show()

s = Simulation()
s.run()
