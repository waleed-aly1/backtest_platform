import pandas as pd
import numpy as np


def calc_moving_linear_regression(ys, columns=['Close'], mlr_prds=10, slope_periods=3, tsize=1.0, descending=True):
    df = ys[columns]
    df.columns = ['ys']
    # Checks to see if the data is ascending
    if sorted(df.index) is not df.index:
        df.sort_index(ascending=True, inplace=True)

    df['xs'] = df.reset_index(drop=True).index + 1
    df['xy'] = df['xs'] * df['ys']
    df['x^2'] = df['xs'] ** 2
    df['y^2'] = df['ys'] ** 2
    df['a'] = (df['ys'].rolling(mlr_prds, min_periods=0).sum() * df['x^2'].rolling(mlr_prds, min_periods=0).sum()
               - df['xs'].rolling(mlr_prds, min_periods=0).sum() * df['xy'].rolling(mlr_prds, min_periods=0).sum()) / \
              (mlr_prds * df['x^2'].rolling(mlr_prds, min_periods=0).sum() - (
                  df['xs'].rolling(mlr_prds, min_periods=0).sum()) ** 2)
    df['b'] = (mlr_prds * (df['xy'].rolling(mlr_prds, min_periods=0).sum()) - df['xs'].rolling(mlr_prds,
                                                                                               min_periods=0).sum() *
               (df['ys'].rolling(mlr_prds, min_periods=0).sum())) / (
                      mlr_prds * df['x^2'].rolling(mlr_prds, min_periods=0).sum()
                      - df['xs'].rolling(mlr_prds, min_periods=0).sum() ** 2)
    df['mlr'] = (df['a'] + (df['xs'] * df['b']))
    df['slope'] = ((df['mlr'] - df['mlr'].shift(slope_periods - 1)) / (
            df['xs'] - df['xs'].shift(slope_periods - 1))) / tsize
    df['mlr'].iloc[0:mlr_prds - 1] = np.nan
    df['slope'].iloc[0:(mlr_prds + slope_periods - 2)] = np.nan
    if descending:
        df = df[::-1]
    df_filtered = df.loc[:, ['ys', 'mlr', 'slope']]
    df_filtered.columns = columns + ['MLR'] + ['Slope']
    return df_filtered


def moving_linear_regression(data, mlr_periods, column=['C'], descending=False):
    df = data[column]
    df.columns = ['ys']
    df['ys'] = df['ys'].replace(r'^\s*$', np.nan, regex=True).fillna(method='ffill')
    df['xs'] = df.reset_index(drop=True).index + 1
    df['xy'] = df['xs'] * df['ys']
    df['x^2'] = df['xs'] ** 2
    df['y^2'] = df['ys'] ** 2
    df['a'] = (df['ys'].rolling(mlr_periods, min_periods=0).sum() * df['x^2'].rolling(mlr_periods, min_periods=0).sum()
               - df['xs'].rolling(mlr_periods, min_periods=0).sum() * df['xy'].rolling(mlr_periods,
                                                                                       min_periods=0).sum()) / \
              (mlr_periods * df['x^2'].rolling(mlr_periods, min_periods=0).sum() - (
                  df['xs'].rolling(mlr_periods, min_periods=0).sum()) ** 2)
    df['b'] = (mlr_periods * (df['xy'].rolling(mlr_periods, min_periods=0).sum()) - df['xs'].rolling(mlr_periods,
                                                                                                     min_periods=0).sum() *
               (df['ys'].rolling(mlr_periods, min_periods=0).sum())) / (
                      mlr_periods * df['x^2'].rolling(mlr_periods, min_periods=0).sum()
                      - df['xs'].rolling(mlr_periods, min_periods=0).sum() ** 2)
    df['mlr'] = (df['a'] + (df['xs'] * df['b']))
    df['mlr'].iloc[0:mlr_periods - 1] = np.nan
    if descending:
        df = df[::-1]
    return df['mlr']


def slope(data, slope_periods, mlr_col_name='MLR', recalc_mlr=False, mlr_periods=None, recalc_column=['C'],
          tick_size=.01, descending=False):
    if recalc_mlr is True:
        df_slope = data[recalc_column]
        df_slope['mlr'] = moving_linear_regression(data, mlr_periods, recalc_column, descending)
    else:
        df_slope = pd.DataFrame()
        df_slope['mlr'] = data[mlr_col_name]

    df_slope['xs'] = df_slope.reset_index(drop=True).index + 1
    df_slope['xy'] = df_slope['mlr'] * df_slope['xs']
    df_slope['x^2'] = df_slope['xs'] ** 2

    df_slope['slope'] = (((df_slope['xy'].rolling(slope_periods, min_periods=0).sum() * slope_periods) -
                          (df_slope['xs'].rolling(slope_periods, min_periods=0).sum() * df_slope['mlr'].rolling(
                              slope_periods, min_periods=0).sum())) / \
                         (slope_periods * df_slope['x^2'].rolling(slope_periods, min_periods=0).sum() - (
                                     df_slope['xs'].rolling(slope_periods, min_periods=0).sum() ** 2))) / tick_size

    first_non_na = df_slope['mlr'].first_valid_index()
    non_na_idx_val = df_slope.index.get_loc(first_non_na)
    df_slope['slope'].iloc[0:(non_na_idx_val + slope_periods - 1)] = np.nan
    if descending:
        df_slope = df_slope[::-1]
    return df_slope['slope']


class MovingLinearRegression:
    def __init__(self, data, column, tick_size=.01, descending=True):
        self.data = data
        self.column = column
        self.tick_size = tick_size
        self.descending = descending

        if sorted(self.data.index) is not self.data.index:
            self.data.sort_index(ascending=True, inplace=True)

    def mlr(self, mlr_periods):
        df = self.data[self.column]
        df.columns = ['ys']
        df['xs'] = df.reset_index(drop=True).index + 1
        df['xy'] = df['xs'] * df['ys']
        df['x^2'] = df['xs'] ** 2
        df['y^2'] = df['ys'] ** 2
        df['a'] = (df['ys'].rolling(mlr_periods, min_periods=0).sum() * df['x^2'].rolling(mlr_periods,
                                                                                          min_periods=0).sum()
                   - df['xs'].rolling(mlr_periods, min_periods=0).sum() * df['xy'].rolling(mlr_periods,
                                                                                           min_periods=0).sum()) / \
                  (mlr_periods * df['x^2'].rolling(mlr_periods, min_periods=0).sum() - (
                      df['xs'].rolling(mlr_periods, min_periods=0).sum()) ** 2)
        df['b'] = (mlr_periods * (df['xy'].rolling(mlr_periods, min_periods=0).sum()) - df['xs'].rolling(mlr_periods,
                                                                                                         min_periods=0).sum() *
                   (df['ys'].rolling(mlr_periods, min_periods=0).sum())) / (
                          mlr_periods * df['x^2'].rolling(mlr_periods, min_periods=0).sum()
                          - df['xs'].rolling(mlr_periods, min_periods=0).sum() ** 2)
        df['mlr'] = (df['a'] + (df['xs'] * df['b']))
        df['mlr'].iloc[0:mlr_periods - 1] = np.nan
        if self.descending:
            df = df[::-1]
        self.mlr = df['mlr']
        return df['mlr']

    def slope(self, mlr_periods, slope_periods):
        df_slope = self.data[self.column]
        df_slope['mlr'] = self.mlr(mlr_periods)
        df_slope['xs'] = df_slope.reset_index(drop=True).index + 1
        df_slope['slope'] = ((df_slope['mlr'] - df_slope['mlr'].shift(slope_periods - 1)) / (
                df_slope['xs'] - df_slope['xs'].shift(slope_periods - 1))) / self.tick_size
        df_slope['slope'].iloc[0:(mlr_periods + slope_periods - 2)] = np.nan
        if self.descending:
            df_slope = df_slope[::-1]
        return df_slope['slope']


"""
Testing area below

data = pd.read_csv('./SampleData/Sample_data.csv', parse_dates=[['Date', 'Time']], index_col=['Date_Time'])
data['mlr'] = moving_linear_regression(data, 10)
data['Slope'] = slope(data, 3, mlr_col_name='mlr')
"""

