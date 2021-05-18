import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Dict, Any, Callable, Iterable
from backtest.time_series_split import TimeSeriesSplit
from backtest.position_logic import position_handler


class Backtest():
    def __init__(self, data, position_logic_function, split: Tuple[any]=None, split_type:str =None, process_data=True,
                 initial_split_method=None, sort_data: str =True, price='O', tick_size=.01, tick_value=10, lot_size=5):
        self.data = data
        self.process_data = process_data
        self.position_logic_function = position_logic_function
        self.price = price
        self.split = split
        self.split_type = split_type
        self.sort_data = sort_data
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.lot_size = lot_size


        if np.logical_or(np.logical_and(self.split is None, initial_split_method is not None),
                                        np.logical_and(self.split is not None, initial_split_method is None)):
            raise ValueError('Please provide Initial Split Type')

        if sort_data:
            if sorted(self.data.index) is not self.data.index:
                self.data.sort_index(ascending=True, inplace=True)

        if self.process_data:
            df = self.data.copy()

            if split_type is None:
                df = position_logic_function.apply(df)
                df['Position'] = list(position_handler(df))
            else:
                if initial_split_method == 'PreSignalCalc':
                    df = TimeSeriesSplit(df).split_selector(split, split_type)
                    df = position_logic_function.apply(df)
                    df['Position'] = list(position_handler(df))
                elif initial_split_method == 'PostSignalCalc':
                    df = position_logic_function.apply(df)
                    df = TimeSeriesSplit(df).split_selector(split, split_type)
                    df['Position'] = list(position_handler(df))
                elif initial_split_method is None:
                    pass
            df.drop(columns=['EntrySignal', 'ExitSignal'], inplace=True)
            self.data = df
        elif not self.process_data:
            self.data = self.data.copy()

    def _trade_count_series(self, split=None, split_type=None):
        trade_series = TimeSeriesSplit(self.data).split_selector(split, split_type)
        trade_series['Trades'] = np.logical_and(trade_series['Position'] != trade_series['Position'].shift(1),
                                                trade_series['Position'] != 0).cumsum()
        trade_series.loc[trade_series['Position'] == 0, 'Trades'] = 0
        return trade_series['Trades']

    def trade_count(self, split=None, split_type=None):
        trade_count = self._trade_count_series(split, split_type)
        return trade_count.max()

    def _entry_price_series(self, split=None, split_type=None):
        entry_price = TimeSeriesSplit(self.data).split_selector(split, split_type)
        if self.price == 'O':
            entry_price['EntryPrice'] = np.where(
                np.logical_and((entry_price['Position'] != entry_price['Position'].shift(1)),
                               entry_price['Position'] != 0), entry_price['O'].shift(1), 0)
        elif self.price == 'C':
            entry_price['EntryPrice'] = np.where(
                np.logical_and((entry_price['Position'] != entry_price['Position'].shift(1)),
                               entry_price['Position'] != 0), entry_price['C'].shift(1), 0)
        return entry_price['EntryPrice']

    def _pnl_series(self, split=None, split_type=None):
        pnl_series = TimeSeriesSplit(self.data).split_selector(split, split_type)
        if self.price == 'O':
            pnl_series['TickChange'] = pnl_series['O'] - pnl_series['O'].shift(1)
        elif self.price == 'C':
            pnl_series['TickChange'] = pnl_series['C'] - pnl_series['C'].shift(1)

        pnl_series['PL'] = (pnl_series['Position'] * pnl_series[
            'TickChange']) / self.tick_size * self.tick_value * self.lot_size
        pnl_series['PL_Accum'] = pnl_series['PL'].cumsum()
        return pnl_series['PL_Accum']

    def pnl(self, split=None, split_type=None):
        pnl = self._pnl_series(split, split_type)
        pnl = pnl.iat[-1]
        return pnl

    def closed_trade_pnl_series(self, split=None, split_type=None):
        closed_trade_pnl = TimeSeriesSplit(self.data).split_selector(split, split_type)
        closed_trade_pnl['PL'] = self._pnl_series(split, split_type)
        closed_trade_pnl['P&L Values'] = np.where(closed_trade_pnl['Position'] - closed_trade_pnl['Position'].shift(-1) != 0, closed_trade_pnl['PL'], np.nan)
        closed_trade_pnl['P&L Values'] = closed_trade_pnl['P&L Values'].fillna(method='ffill')
        closed_trade_pnl['Closed P&L'] = np.where(closed_trade_pnl['Position'] - closed_trade_pnl['Position'].shift(-1) != 0,
                                                closed_trade_pnl['P&L Values'] - closed_trade_pnl['P&L Values'].shift(1), 0)
        return closed_trade_pnl['Closed P&L']

    def _closed_max_draw_series(self, split=None, split_type=None):
        max_draw = TimeSeriesSplit(self.data).split_selector(split, split_type)
        max_draw['PL_Accum'] = self._pnl_series(split, split_type)
        max_draw['Max'] = max_draw['PL_Accum'].cummax()
        max_draw['Diff'] = max_draw['PL_Accum'] - max_draw['Max']
        max_draw['PeriodDraw_Closed'] = max_draw.groupby(max_draw['Diff'].eq(0).cumsum())['PL_Accum'] \
                                             .transform(lambda x: x.cummin()) - max_draw['Max']
        return max_draw['PeriodDraw_Closed']

    def closed_max_draw(self, split=None, split_type=None):
        max_draw = self._closed_max_draw_series(split, split_type).min()
        return max_draw

    def open_max_draw(self, split=None, split_type=None):
        open_draw = TimeSeriesSplit(self.data.copy()).split_selector(split, split_type)
        open_draw['Trades'] = self._trade_count_series(split, split_type)
        conditions = [open_draw['Position'] == 1, open_draw['Position'] == -1]
        long = open_draw.groupby('Trades')['L'].cummin() - open_draw.groupby('Trades')['EntryPrice'].cummax()
        short = open_draw.groupby('Trades')['EntryPrice'].cummax() - open_draw.groupby('Trades')['H'].cummax()
        values = [long, short]
        open_draw['MaxOpenDraw_Pd'] = np.select(conditions, values, 0)
        return open_draw['MaxOpenDraw_Pd'].min() / self.tick_size * self.lot_size * self.tick_value

    def profit_to_max_draw(self, split=None, split_type=None):
        pmd = self.pnl(split, split_type) / abs(self.closed_max_draw(split, split_type))
        return pmd

    def _average_win_series(self, split=None, split_type=None):
        avg_win = pd.DataFrame()
        avg_win['PL'] = self.closed_trade_pnl_series(split, split_type)
        avg_win['AverageWin'] = avg_win[avg_win['PL'].gt(0)]['PL'].expanding().mean()
        avg_win['AverageWin'] = avg_win['AverageWin'].ffill()
        return avg_win['AverageWin']

    def average_win(self, split, split_type=None):
        average_win = self._average_win_series(split, split_type).iat[-1]
        return average_win

    def _average_loss_series(self, split=None, split_type=None):
        avg_loss = pd.DataFrame()
        avg_loss['PL'] = self.closed_trade_pnl_series(split, split_type)
        avg_loss['AverageLoss'] = avg_loss[avg_loss['PL'].lt(0)]['PL'].expanding().mean()
        avg_loss['AverageLoss'] = avg_loss['AverageLoss'].ffill()
        return avg_loss['AverageLoss']

    def average_loss(self, split=None, split_type=None):
        average_loss = self._average_loss_series(split, split_type).iat[-1]
        return average_loss

    def max_win(self, split=None, split_type=None):
        return self._pnl_series(split, split_type).max()

    def max_loss(self, split=None, split_type=None):
        return self._pnl_series(split, split_type).min()

    def long_trades(self, split=None, split_type=None):
        longs = TimeSeriesSplit(self.data.copy()).split_selector(split, split_type)
        longs = np.logical_and(longs['Position'] == 1, np.logical_not(longs['Position'].shift(1) == 1)).sum()
        return longs

    def short_trades(self, split=None, split_type=None):
        shorts = TimeSeriesSplit(self.data.copy()).split_selector(split, split_type)
        shorts = np.logical_and(shorts['Position'] == -1, np.logical_not(shorts['Position'].shift(1) == -1)).sum()
        return shorts

    def output_all_metrics_as_df(self, split=None, split_type=None):
        output_df = TimeSeriesSplit(self.data.copy()).split_selector(split, split_type)
        output_df['Trades'] = self._trade_count_series(split, split_type)
        output_df['EntryPrice'] = self._entry_price_series(split, split_type)
        output_df['PL'] = self._pnl_series(split, split_type)
        output_df['InTradePL_Accum'] = output_df['PL'].cumsum()
        output_df['ClosedPL'] = self.closed_trade_pnl_series(split, split_type)
        output_df['ClosedPL_Accum'] = output_df['ClosedPL'].cumsum()
        output_df['Closed_Pnl_CumSum'] = output_df['ClosedPL'].cumsum()
        output_df['Closed_Pnl_CumMax'] = output_df['Closed_Pnl_CumSum'].cummax()
        output_df['ClosedTradeDrawDown'] = output_df['Closed_Pnl_CumSum']-output_df['Closed_Pnl_CumMax']
        output_df['MaxClosedDraw'] = self._closed_max_draw_series(split, split_type)
        output_df['Max'] = output_df['PL'].cummax()
        output_df['MaxDrawPerPeriod'] = np.where(output_df['Max'] != output_df['Max'].shift(-1), output_df['MaxClosedDraw'],0)
        output_df['AverageWin'] = self._average_win_series(split, split_type)
        output_df['AverageLoss'] = self._average_loss_series(split, split_type)
        output_df.drop(['Max','Closed_Pnl_CumSum', 'Closed_Pnl_CumMax'], axis=1, inplace=True)
        return output_df

    def agg_stats(self, split=None, split_type=None):
        stats_df = pd.DataFrame()
        agg_stats = self._pnl_series(split, split_type)
        trade_series = self._trade_count_series(split, split_type)

        # Calculates the Max Drawdown info within the method so as not to reuse the pl series
        stats_df['TradeCount'] = trade_series
        stats_df['PL_Accum'] = agg_stats
        stats_df['Max'] = stats_df['PL_Accum'].cummax()
        stats_df['Diff'] = stats_df['PL_Accum'] - stats_df['Max']

        stats_df['PeriodDraw_Closed'] = stats_df.groupby(stats_df['Diff'].eq(0).cumsum())['PL_Accum'] \
                             .transform(lambda x: x.cummin()) - stats_df['Max']

        stats_df['MaxPerPeriod'] = np.where(stats_df['Max'] != stats_df['Max'].shift(-1), stats_df['PeriodDraw_Closed'],0)

        #Calculate Closed Pl
        stats_df['P&L Values'] = np.where(stats_df['TradeCount'] - stats_df['TradeCount'].shift(-1) != 0, stats_df['PL_Accum'], np.nan)
        stats_df['P&L Values'] = stats_df['P&L Values'].fillna(method='ffill')
        stats_df['Closed P&L'] = np.where(stats_df['TradeCount'] - stats_df['TradeCount'].shift(-1) != 0,
                                          stats_df['P&L Values'] - stats_df['P&L Values'].shift(1), 0)

        # Calulcate Closed Trade DrawDown
        stats_df['Closed P&L CumSum'] = stats_df['Closed P&L'].cumsum()
        stats_df['Closed P&L CumMax'] = stats_df['Closed P&L CumSum'].cummax()
        stats_df['ClosedTradeDrawDown'] = stats_df['Closed P&L CumSum']-stats_df['Closed P&L CumMax']

        # Calculate Closed Trade Max Draw Down
        stats_df['Max_Closed'] = stats_df['Closed P&L'].cummax()



        long_trades = self.long_trades(split, split_type)
        short_trades = self.short_trades(split, split_type)

        stats = {
            'Total Net Profit': agg_stats.iat[-1],
            'Total Trades': trade_series.max(),
            'In Trade Max Draw Down(Closed Bar)': stats_df['PeriodDraw_Closed'].min(),
            'Profit to Max Draw': agg_stats.iat[-1] / abs(stats_df['PeriodDraw_Closed'].min()),
            'Closed Trade Max Draw': stats_df['ClosedTradeDrawDown'].min(),
            'Profit to Closed Trade Max Draw': agg_stats.iat[-1] / abs(stats_df['ClosedTradeDrawDown'].min()),
            'PercentWinners': round((stats_df['Closed P&L'].gt(0).sum() / stats_df['Closed P&L'].ne(0).sum()) * 100, 2),
            'DrawPeriods Over -30k': round(stats_df['MaxPerPeriod'].lt(-30000).sum(),0),
            'Average Draw Over -30k': round(stats_df[stats_df['MaxPerPeriod'].lt(-30000)]['MaxPerPeriod'].mean(),0),
            'Draw Periods Over -10k': round(stats_df['MaxPerPeriod'].lt(-10000).sum(),0),
            'Average Draw Over -10k': round(stats_df[stats_df['MaxPerPeriod'].lt(-10000)]['MaxPerPeriod'].mean(),0),
            'MaxWin': round(stats_df['Closed P&L'].max(),0),
            'MaxLoss': round(stats_df['Closed P&L'].min(),0),
            'Avg Win': round(stats_df[stats_df['Closed P&L'].gt(0)]['Closed P&L'].mean(),0),
            'Avg Loss': round(stats_df[stats_df['Closed P&L'].lt(0)]['Closed P&L'].mean(),0),
            'Long Trades': long_trades,
            'Short Trades': short_trades,
            'Percent Long': round(long_trades/trade_series.max()*100, 2)
        }
        return stats

    def _grouping_function(self, groupby_data):
        agg_dict = {'EntryTime': groupby_data.iloc[0,].name,
                    'TradeType': groupby_data['Position'].iat[0],
                    'EntryPrice': groupby_data['EntryPrice'].max(),
                    'ExitTime': groupby_data.iloc[-1,].name,
                    'ExitPrice': groupby_data['O'].iat[-1]
                    }

        return pd.Series(agg_dict, index=['EntryTime', 'ExitTime', 'TradeType', 'EntryPrice', 'ExitPrice'])

    def tabular_trades(self, group_by_col='Trades', split=None, split_type=None):
        grouping_df = TimeSeriesSplit(self.data).split_selector(split, split_type)
        grouping_df['Trades'] = self._trade_count_series()
        grouping_df['EntryPrice'] = self._entry_price_series()

        compact_results = grouping_df.groupby([group_by_col]).apply(self._grouping_function)

        compact_results['PL'] = (compact_results['ExitPrice'] - compact_results['EntryPrice']) * compact_results['TradeType']\
                                /self.tick_size * self.tick_value * self.lot_size
        compact_results['TotPL'] = compact_results['PL'].cumsum()
        compact_results['Max'] = compact_results['TotPL'].cummax()
        compact_results['Diff'] = compact_results['TotPL'] - compact_results['Max']
        df = compact_results.reset_index()
        m = []
        for i in df.index:
            if df.iloc[i, df.columns.get_loc('TotPL')] == df.iloc[i, df.columns.get_loc('Max')]:
                m.append(df.iloc[i, df.columns.get_loc('Diff')])
            elif i == 0:
                m.append(0)
            else:
                m.append(min(m[i - 1], df.iloc[i, df.columns.get_loc('Diff')]))

        compact_results['PeriodDraw'] = m
        compact_results['MaxDraw'] = compact_results['PeriodDraw'].cummin()
        compact_results['PMD'] = compact_results['TotPL'] / compact_results['MaxDraw'].abs()
        compact_results['PercentWin'] = ((compact_results['PL'] > 0).cumsum() / compact_results.index.values) * 100
        compact_results['Day'] = compact_results['EntryTime'].dt.day
        compact_results['Month'] = compact_results['EntryTime'].dt.month
        compact_results['Year'] = compact_results['EntryTime'].dt.year
        return compact_results