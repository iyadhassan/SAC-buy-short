import yfinance as yf
import numpy as np
import pandas as pd

class DataHandler:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._fetch_data()
        self.train_test_split()

    def _fetch_data(self):
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError(f"No data available for {self.symbol} between {self.start_date} and {self.end_date}")
        data = data.reset_index()
        data.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        return data

    def train_test_split(self, test_size=0.2):
        split_index = int(len(self.data) * (1 - test_size))
        self.train_data = self.data.iloc[:split_index]
        self.test_data = self.data.iloc[split_index:]

        # Check for NaN values
        if self.train_data.isnull().values.any() or self.test_data.isnull().values.any():
            print("NaN values detected in the data. Filling with forward fill method.")
            self.train_data = self.train_data.fillna(method='ffill')
            self.test_data = self.test_data.fillna(method='ffill')

    def get_train_data(self):
        return self.train_data.to_dict('records')

    def get_test_data(self):
        return self.test_data.to_dict('records')

    def get_normalized_data(self):
        normalized_data = self.data.copy()
        for column in ['open', 'high', 'low', 'close', 'adj_close']:
            normalized_data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()
        return normalized_data