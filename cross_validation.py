import pandas as pd
from position_logic import PositionLogicRange, PositionLogic
from typing import List, Tuple, Iterator, Dict, Any, Callable, Iterable
import time
from time_series_split import TimeSeriesSplit
from backtester import Backtest
import matplotlib.pyplot as plt
from mpldatacursor import datacursor


class CrossValidate():
    def __init__(self, data:pd.DataFrame, position_logic_range: PositionLogicRange, split: Tuple[int, int], timeseries_split:Tuple[str, str] = None, timeseries_split_type:str=None,
                 tick_size=.01, tick_value=10, lot_size=5):
        self.data = data
        self.no_split = split[0]
        self.initial_split_multiple = split[1]
        self.time_series_split = timeseries_split
        self.time_series_split_type = timeseries_split_type
        self.position_logic_range = position_logic_range
        self.position_logic_combos = list(position_logic_range.position_logic_generator())
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.lot_size = lot_size

        if split is None:
            pass
        else:
            self.data = TimeSeriesSplit(self.data).split_selector(self.time_series_split, self.time_series_split_type)

    def _multi_index_to_position_logic(self, multi_index: Tuple[Tuple[int,...], ...]) -> PositionLogic:
        """
        Auxiliar function that produces the PositionLogic corresponding to the given MultiIndex of the walk forward cross-validation.
        :param multi_index: A tuple of tuples containing the signal function parameters and the entry and exit function parameters.
        :return: The PositionLogic corresponding to the MultiIndex.
        """
        exit_parameters_tuple = multi_index[-1]
        entry_parameters_tuple = multi_index[-2]
        signal_function_list_params = multi_index[:-2]
        return self.position_logic_range.fit_to_parameters(signal_function_list_params, entry_parameters_tuple, exit_parameters_tuple)

    def walk_forward(self, optimzation_window_type = 'RollingOptimizationWindow'):
        optimization_results = {}
        walk_forward_results = {}

        for e, position_logic in enumerate(self.position_logic_combos):
            start = time.time()
            param_list = tuple(signal_function.function_parameters_tuple for signal_function in position_logic.signal_func_list)\
                         + position_logic.entry.function_parameters_tuple + position_logic.exit.function_parameters_tuple
            backtest_instance = Backtest(self.data, position_logic, tick_size=self.tick_size, tick_value=self.tick_value, lot_size=self.lot_size)
            opt_loop_results = {}
            wf_loop_results = {}
            for i in range(1, self.no_split+1):
                opt_loop_results[i] = backtest_instance.profit_to_max_draw(split=(self.no_split, self.initial_split_multiple, i), split_type=optimzation_window_type)
                wf_loop_results[i] = backtest_instance.agg_stats(split=(self.no_split, self.initial_split_multiple, i), split_type='WalkForwardWindow')
                print(f"Finished window {i} of {self.no_split}")
            optimization_results[str(param_list)] = opt_loop_results
            walk_forward_results[str(param_list)] = wf_loop_results

            end = time.time()
            print(f"Run {e+1}/{len(self.position_logic_combos)} ")
            print(f"{round(((len(self.position_logic_combos)-e+1)*(end-start))/60,2)}min Remaining")
            print(round(end-start,3))

        opt_df = pd.DataFrame.from_dict(optimization_results, orient='index')
        wf_df = pd.DataFrame.from_dict(walk_forward_results, orient='index')

        testdict = {}
        for i in opt_df.columns:
            test_dates = TimeSeriesSplit(self.data).walk_forward_dates((self.no_split, self.initial_split_multiple, i))
            top_param = opt_df[i].idxmax(axis=0)
            dict_val = wf_df.at[top_param, i]
            testdict[i] = dict_val
            testdict[i]['Params'] = top_param
            testdict[i]['WindowStart'] = test_dates[0]
            testdict[i]['WindowEnd'] = test_dates[1]
        final_df = pd.DataFrame.from_dict(testdict, orient='index') # This should be outside the loop

        return final_df

def plot_equity_curve(data):
    fig, ax = plt.subplots()
    # ax.bar(x=stats_df.index, height=stats_df['MaxDrawPerPeriod'],width = 10, color='r')
    ax.step(data.index, data['MaxDrawPerPeriod'], color='r', alpha=.2)
    axtwin = ax.twinx()
    axtwin.plot(data.index, data['ClosedPL_Accum'])
    ax.set_ylim(0, -300000)
    ax.set_ylabel('Drawdown', color='r')
    axtwin.set_ylabel('P&L')
    datacursor()
    fig.tight_layout()
    return plt.show()