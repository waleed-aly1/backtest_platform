import numpy as np
import pandas as pd
import datetime
from itertools import product
from typing import List, Tuple, Iterator, Dict, Any, Callable, Iterable



def entry_signal_function(data, entry_parameters_tuple, offset=True):
    """"
    calculates the entry signal
    Will say either Entry Signal or blank
    :param signal_series: data used to calculate the entry series
    :param offset: If set to true, will return Entrysignal when the previous bar is an entry signal at the close
    """



    if offset == True:
        l_entry_off = np.logical_and(data['Slope'].shift(1) > 0, data['Slope'].shift(2) < 0)
        s_entry_off = np.logical_and(data['Slope'].shift(1) < 0, data['Slope'].shift(2) > 0)
        conditions1 = [l_entry_off, s_entry_off]

    else:
        l_entry = np.logical_and(data['Slope'] > 0, data['Slope'].shift(1) < 0)
        s_entry = np.logical_and(data['Slope'] < 0, data['Slope'].shift(1) > 0)
        conditions1 = [l_entry, s_entry]
    values1 = ['LongEntry', 'ShortEntry']
    return np.select(conditions1, values1, '')


def exit_signal_function(data, exit_parameters_tuple, offset=True):
    """"
    calculates whether a bar qualifies as an exit condition regardless of whether you are in a trade or not
    Will say either Exit or blank
    :param data: data used for the calculation
    :param exit_parameters_tuple: parameters passed in as exit or entry conditions
    :param offset: If set to true, will return EntrySignal when the previous bar is an entry signal at the close
    """



    if offset:
        # long exit conditions
        slope_xbelow_0 = np.logical_and(data['Slope'].shift(1) < 0, data['Slope'].shift(2) > 0)


        # l_exit = np.logical_or(slope_xbelow_0, upper_bound_slope)

        # short exit conditions
        slope_xabove_0 = np.logical_and(data['Slope'].shift(1) > 0, data['Slope'].shift(2) < 0)
        # lower_bound_slope = np.logical_and(data['Slope'].shift(1) < -exit_parameters_tuple[0], data['Slope'].shift(2) > -exit_parameters_tuple[0])

        # s_exit = np.logical_or(slope_xabove_0, lower_bound_slope)

    else:
        pass
    conditions2 = [slope_xbelow_0, slope_xabove_0]
    values2 = ['LongExit', 'ShortExit']
    return np.select(conditions2, values2, '')

class SignalFunction():
    """
    Class that contains the function to compute the signal from input data.
    """

    def __init__(self, signal_function: Callable[..., pd.DataFrame], signal_function_parameters_tuple, signal_function_kwarg_params=None, name=None):
        self.signal_function = signal_function
        self.name = name
        self.function_parameters_tuple = signal_function_parameters_tuple
        if signal_function_kwarg_params is None:
            self.sig_kwargs_dict = {}
        else:
            self.sig_kwargs_dict = signal_function_kwarg_params

    def apply(self, input_data):
        """
        Applies the signal function into the given input data.
        :return: A pandas series where the signal has been
        """
        return self.signal_function(input_data, *self.function_parameters_tuple, **self.sig_kwargs_dict)


class SignalFunctionRange():
    """
    Class defining a range of SignalFunction.
    """
    def __init__(self, signal_function: Callable[..., pd.DataFrame], parameters_range_tuple: Tuple[Iterable[int], ...], kwarg_params: Dict[str, Any]=None, name: str=None):
        self.name = name
        self.signal_function = signal_function
        self.parameters_range_tuple = parameters_range_tuple
        if kwarg_params is None:
            self.sig_kwarg_dict = {}
        else:
            self.sig_kwarg_dict = kwarg_params

    def fit_to_parameters(self, parameters_tuple) -> SignalFunction:
        """
        Given a specific tuple of parameters it produces the corresponding SignalFunction.
        :param parameters_tuple: A tuple of parameters used by the signal function.
        :return: A SignalFunction that uses the specified parameters_tuple.
        """
        return SignalFunction(self.signal_function, parameters_tuple, self.sig_kwarg_dict, self.name)

    def signal_function_generator(self) -> Iterator[SignalFunction]:
        """
        :return: Generator that produces all SignalFunctions possible of the given parameters_range_tuple.
        """
        return (self.fit_to_parameters(parameters_tuple) for parameters_tuple in product(*self.parameters_range_tuple))



class PositionFunction():
    """
    Class that contains the logic defining a position decision function and its parameters.
    """

    def __init__(self, position_function, function_parameters_tuple, offset=True):
        self.position_function = position_function
        self.function_parameters_tuple = function_parameters_tuple
        self.offset = offset

    def apply(self, input_data):
        """
        Applies the position logic into the given signal_series.
        :return: A pandas series where the signal has been
        """
        return self.position_function(input_data, self.function_parameters_tuple, self.offset)


class PositionFunctionRange():
    """
    Class defining a range of PositionFunction.
    """
    def __init__(self, position_function, parameters_range_tuple, offset):
        self.position_function = position_function
        self.parameters_range_tuple = parameters_range_tuple
        self.offset = offset

    def fit_to_parameters(self, parameters_tuple) -> PositionFunction:
        return PositionFunction(self.position_function, parameters_tuple, self.offset)

    def position_function_generator(self) -> Iterator[PositionFunction]:
        """
        :return: Generator that produces all PositionFunctions possible of the given parameters_range_tuple.
        """
        return (self.fit_to_parameters(parameters_tuple) for parameters_tuple in product(*self.parameters_range_tuple))


class PositionLogic():
    """
    Class that contains the logic defining the entry and exit positions.
    :param signal_function_list: a list of object SignalFunctions that contain the function,params,keyword params used for calculating any  calculated fileds used in the strategy
    :param entry_function: a function containing the logic which defines an entry signal
    :param entry_parameters_tuple: a tuple of the parameters used in the entry function
    :param exit_function: a function containing the logic which defines an exit signal
    :param exit_parameters_tuple: a tuple of the parameters used in the exit function
    :param offset: boolean expression passed to the entry and exit functions which determines if the 1,-1,0 will offset by 1 bar (mainly used for trades only layout of the backtester)
    """

    def __init__(self, signal_function_list: List[SignalFunction], entry_function: PositionFunction, exit_function: PositionFunction):
        self.signal_func_list = signal_function_list
        self.entry = entry_function
        self.exit = exit_function

    def apply(self, data):
        for i, signal_function in enumerate(self.signal_func_list):
            if signal_function.name is None:
                data[str(i)] = signal_function.apply(data)
            else:
                data[signal_function.name] = signal_function.apply(data)
        data['EntrySignal'] = self.entry.apply(data)
        data['ExitSignal'] = self.exit.apply(data)
        return data



class PositionLogicRange():
    """
    Class deifning a range of PositionLogics.
    """

    def __init__(self, signal_function_range_list: List[SignalFunctionRange], entry_function_range: PositionFunctionRange, exit_function_range: PositionFunctionRange):
        self.signal_function_list = signal_function_range_list
        self.entry = entry_function_range
        self.exit = exit_function_range

    def _signal_function_combinations(self) -> Iterator[Iterator[SignalFunction]]:
        """
        :return: Generates all possible combinations of signal_function_lists from the specified ranges.
        """
        return product(*(signal_function_range.signal_function_generator() for signal_function_range in self.signal_function_list))

    def position_logic_generator(self) -> Iterator[PositionLogic]:
        """
        :return: Generates all PositionLogis compatible with the given list of StrategyFunctionRanges, the entry PositionFunctionRange and the
        exit PositionFunctionRange.
        """
        return (PositionLogic(signal_function_list, entry_function, exit_function) for signal_function_list, entry_function, exit_function in
                product(self._signal_function_combinations(), self.entry.position_function_generator(), self.exit.position_function_generator()))

    def fit_to_parameters(self, signal_function_parameters_list: Iterable[Tuple[int, ...]], entry_parameters: Tuple[int, ...], exit_parameters: Tuple[int, ...]) \
            -> PositionLogic:
        """
        Produces a specific PositionLogic from the PositionLogic range of the given signal_function_parameters_list, entry_parameters and exit_parameters.
        :param signal_function_parameters_list: A list of tuples containing the parameters of each of signal_function.
        :param entry_parameters: A tuple of parameters for the entry function.
        :param exit_parameters: A tuple of parameters for the exit function.
        :return: The PositionLogic of the given parameters.
        """
        signal_functions = [signal_function.fit_to_parameters(parameters)
                            for signal_function, parameters in zip(self.signal_function_list, signal_function_parameters_list)]
        entry_function = self.entry.fit_to_parameters(entry_parameters)
        exit_function = self.exit.fit_to_parameters(exit_parameters)
        position_logic = PositionLogic(signal_functions, entry_function, exit_function)
        return position_logic



def position_handler(data: pd.DataFrame) -> Iterator[int]:
    """""
    assigns numerical position values based on entry exit conditions
    kept entry and exit positions for flexibility. Added the offset feature to easily switch between close of signal bar vs open of the next bar
    0,1,-1 = flat, long, short
    :param signal_series: data used to calculate the entry and exit signal
    :param position_logic: A PositionLogic object defining the functions and parameters to use to determine the entry and exit signals.
    :return iterator of positional series
    """

    curr_pos = 0
    for i in data.reset_index().index:
        if i == 0:
            curr_pos = 0
            yield 0
        elif data.iloc[i - 1, data.columns.get_loc('EntrySignal')] == 'LongEntry':
            curr_pos = 1
            yield 1
        elif np.logical_and(curr_pos == 1, data.iloc[i - 1, data.columns.get_loc('ExitSignal')] != 'LongExit'):
            curr_pos = 1
            yield 1
        elif data.iloc[i - 1, data.columns.get_loc('EntrySignal')] == 'ShortEntry':
            curr_pos = -1
            yield -1
        elif np.logical_and(curr_pos == -1, data.iloc[i - 1, data.columns.get_loc('ExitSignal')] != 'ShortExit'):
            curr_pos = -1
            yield -1
        else:
            curr_pos = 0
            yield 0
