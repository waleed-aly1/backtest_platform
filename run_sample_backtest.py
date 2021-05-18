import pandas as pd
from backtest.position_logic import PositionLogic, PositionFunction, entry_signal_function, exit_signal_function, \
    SignalFunction, PositionLogicRange, PositionFunctionRange, SignalFunctionRange
from backtest.optimizer import Optimizer
import backtest.moving_linear_regression as mlr
from backtest.cross_validation import plot_equity_curve, CrossValidate
from backtest.backtester import Backtest

rbe1597 = pd.read_excel('./data/rbe1597.xlsx', index_col='DateTime')

mlr_signal_function = SignalFunction(mlr.moving_linear_regression, (40,), {'descending': False}, name='MLR')
slope_signal_function = SignalFunction(mlr.slope, (4,), {'descending': False}, name='Slope')
entry_func = PositionFunction(entry_signal_function, ())
exit_func = PositionFunction(exit_signal_function, ())

position_logic_example = PositionLogic([mlr_signal_function, slope_signal_function], entry_func, exit_func)

print('Starting Backtest Instantiation')

backtester_example = Backtest(rbe1597, position_logic_example, split=('1-1-2012', '12-31-2018'), split_type='DateRange',
                              initial_split_method='PostSignalCalc', tick_size=1, tick_value=4.2)

print('Running Output DF')
aggstats = backtester_example.agg_stats()
stats_df = backtester_example.output_all_metrics_as_df()

print('Running Tabular Trades')

compact_data = backtester_example.tabular_trades()
print('FinishedTabularTrades')

stats_df.to_csv('BackTestResultsExampleOutput.csv')
compact_data.to_csv('BackTestResultsCompactOutput.csv')

agg_stats_df = pd.DataFrame.from_dict(aggstats, orient='index')
agg_stats_df.to_csv('BackTestResultsAggStats.csv')

# Optimization area
print('Starting Optimiaztion')

mlr_range = SignalFunctionRange(mlr.moving_linear_regression, (range(5,100,5),), {'descending':False},name='MLR')
slope_range = SignalFunctionRange(mlr.slope, (range(2,11),), {'descending':False},name='Slope')
entry_range = PositionFunctionRange(entry_signal_function,(), offset=True)
exit_range = PositionFunctionRange(exit_signal_function,(), offset=True)
mlr_cross_holder = PositionLogicRange(signal_function_range_list=[mlr_range, slope_range],
                                      entry_function_range=entry_range, exit_function_range=exit_range)
mlr_cross_strategy_logic = list(mlr_cross_holder.position_logic_generator())

opt_results = Optimizer(rbe1597, split=('06-1-2014', '12-9-2018'),
                        split_type='DateRange', tick_size=1, tick_value=4.2).sorted_optimization(mlr_cross_strategy_logic, split=('1-1-2016','12-31-2018'), split_type='DateRange')
opt_results.to_csv('OptimzedResultsOutputExample.csv')


wf_df2 = CrossValidate(rbe1597, mlr_cross_strategy_logic, (30, 2),timeseries_split=('12-16-2015','12-9-2018'), timeseries_split_type='DateRange', tick_size=1, tick_value=4.2).walk_forward()

wf_df2.to_csv('WalkForwardResultsExampleOutput.csv')

plot_equity_curve(stats_df)
print('done')
