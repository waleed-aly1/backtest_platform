import pandas as pd
from backtester import Backtest
from time_series_split import TimeSeriesSplit
import time

class Optimizer():
    def __init__(self, data, split=None, split_type=None, tick_size=.01, tick_value=10, lot_size=5):
        self.data = data
        self.split = split
        self.split_type = split_type
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.lot_size = lot_size

        if isinstance(self.data,Backtest):
            self.data = data.data
        else:
            if split is None:
                pass
            else:
                self.data = TimeSeriesSplit(self.data).split_selector(self.split, self.split_type)

    def _optimize(self, combo_logic, split=None, split_type=None):
        results = {}
        length = len(combo_logic)
        for i, position_logic in enumerate(combo_logic):
            start = time.time()
            stats = Backtest(self.data, position_logic, tick_size=self.tick_size, tick_value=self.tick_value, lot_size=self.lot_size).agg_stats(split, split_type)

            param_tuple = tuple(signal_function.function_parameters_tuple for signal_function in position_logic.signal_func_list) \
            + position_logic.entry.function_parameters_tuple + position_logic.exit.function_parameters_tuple

            results[param_tuple] = stats
            # results[position_logic.signal_function.function_parameters_tuple,
            #     position_logic.entry.function_parameters_tuple, position_logic.exit.function_parameters_tuple] = stats
            end = time.time()
            print(f"{i+1}/{length}")
            print(f"{round(((length-i+1)*(end-start))/60,0)}min Remaining")
            print(round(end-start,3))
        df = pd.DataFrame.from_dict(results, orient='index')
        return df

    def sorted_optimization(self, combo_logic, sort_col='Profit to Max Draw', split=None, split_type=None):
        df_sorted = self._optimize(combo_logic, split, split_type)
        df_sorted.sort_values(by=[sort_col], inplace=True, ascending=False)
        return df_sorted

    def top_result(self, combo_logic, top_column='Profit to Max Draw', split=None, split_type=None):
        top_param = self._optimize(combo_logic, split, split_type)
        index_val = top_param[top_column].idxmax()
        return index_val
