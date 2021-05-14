import numpy as np
import pandas as pd
import datetime
from typing import List, Tuple, Iterator, Dict, Any, Callable, Iterable



class TimeSeriesSplit():
    def __init__(self, data: pd.DataFrame, sort_data=False):
        self.data = data
        self.sort_data = sort_data

        if self.sort_data:
            if sorted(self.data.index) is not self.data.index:
                self.data.sort_index(ascending=True, inplace=True)

    def date_range_split(self, split) -> pd.DataFrame:
        """
        :param split: a tuple in datetime format that contains the start date and the end date of the range. Year-month-day
        :return: dataframe that contains only the data within the date range
        """

        date_range = self.data.copy()

        nearest_start = date_range.iloc[
            date_range.index.get_loc(datetime.datetime.strptime(split[0], '%m-%d-%Y'), method='nearest')].name
        nearest_end = date_range.iloc[
            date_range.index.get_loc(datetime.datetime.strptime(split[1], '%m-%d-%Y'), method='nearest')].name
        date_range = date_range.truncate(before=nearest_start, after=nearest_end)
        return date_range

    def exclude_date_range(self, split: Tuple[str, str]) -> pd.DataFrame:
        """
        :param split: a tuple in datetime format that contains the start date and the end date of the range to be excluded. Year-month-day
        :return: dataframe that removes the dates passed in and returns the resulting dataframe.
        """
        ex_date_range = self.data.copy()
        nearest_start = ex_date_range.iloc[ex_date_range.index.get_loc(datetime.datetime.strptime(split[0], '%m-%d-%Y'),
                                                                       method='nearest')].name
        nearest_end = ex_date_range.iloc[ex_date_range.index.get_loc(datetime.datetime.strptime(split[1], '%m-%d-%Y')
                                                                     , method='nearest')].name
        ex_date_range = ex_date_range.reset_index()
        ex_date_range = ex_date_range.where(np.logical_or(ex_date_range['Date_Time'] < nearest_start,
                                                          ex_date_range['Date_Time'] > nearest_end))
        ex_date_range = ex_date_range.set_index('Date_Time').dropna()
        return ex_date_range

    def step_size(self, split: Tuple[int, int, int]) -> int:
        """
        calculates the step size used in the walk forward test
        :param split: Tuple containing 3 integers. first is total number of windows desired, second is initial
                       window multiple, third is the window number who's indices are to be returned
        :return: integer that is the size of the step used for each window in walk forward test
        """
        no_splits = split[0]
        initial_window_multiple = split[1]
        step_size = int(np.floor(self.data.shape[0] / (initial_window_multiple + no_splits)))
        return step_size

    def initial_window_size(self, split: Tuple[int, int, int]) -> int:
        """
        calculates the initial window size that will be used for the initial optimization. uses the multiple in the 2nd
        position of the split tuple
        :param split: Tuple containing 3 integers. first is total number of windows desired, second is initial
                      window multiple, third is the window number who's indices are to be returned
        :return: integer that is the size of the initial optimization window
        """
        initial_window_multiple = split[1]
        initial_window_size = int(np.floor(self.step_size(split) * initial_window_multiple))
        return initial_window_size

    def optimization_indices(self, split: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        returns the optimization indices of each of the optimization windows
        :param split: Tuple containing 3 integers. first is total number of windows desired, second is initial
                      window multiple, third is the window number who's indices are to be returned
        :return: tuple with start index number and end index number
        """
        no_splits = split[0]
        return_split = split[2]
        start = 0
        if no_splits == return_split:
            end = self.data.shape[0]
        else:
            end = (return_split-1 * self.step_size(split)) + self.initial_window_size(split)
        return start, end-1

    def optimization_window(self, split: Tuple[int, int, int]) -> pd.DataFrame:
        """
        :param split: Tuple containing 3 integers. first is total number of windows desired, second is initial
                      window multiple, third is the window number who's indices are to be returned
        :return: dataframe containing the desired optimization window
        """
        opt_df = self.data.copy()
        opt_df = opt_df[self.optimization_indices(split)[0]: self.optimization_indices(split)[1]+1]
        return opt_df

    def rolling_optimization_indices(self, split: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        returns the optimization indices of each of the optimization windows
        :param split: Tuple containing 3 integers. first is total number of windows desired, second is initial
                      window multiple, third is the window number who's indices are to be returned
        :return: tuple with start index number and end index number
        """

        return_split = split[2]
        step_size = self.step_size(split)
        initial_window_size = self.initial_window_size(split)

        start = max((return_split - 1) * step_size - 1, 0)
        end = ((return_split-1) * step_size) + initial_window_size


        return start, end-1

    def rolling_optimization_window(self, split: Tuple[int, int, int]) -> pd.DataFrame:
        """
        moves the optimzization window forward in each window vs starting at 0
        :param split: Tuple containing 3 integers. first is total number of windows desired, second is initial
                      window multiple, third is the window number who's indices are to be returned
        :return: dataframe containing the desired optimization window
        """
        opt_df = self.data.copy()
        opt_df = opt_df[self.rolling_optimization_indices(split)[0]: self.rolling_optimization_indices(split)[1]+1]
        return opt_df


    def walk_forward_indices(self, split: Tuple[int, int, int])-> Tuple[int, int]:
        """
        :param split: Tuple containing 3 integers. first is total number of windows desired, second is initial
        window multiple, third is the window number who's indices are to be returned
        :return: tuple with start index number and end index number
        """
        no_splits = split[0]
        step_size = self.step_size(split)
        initial_window_size = self.initial_window_size(split)
        return_split = split[2]

        if return_split > no_splits:
            raise ValueError('Return Split Greater than Total Number of Splits')

        if return_split == 0:
            start = initial_window_size
        else:
            start = ((return_split - 1) * step_size) + initial_window_size + 1

        if no_splits == return_split:
            end = self.data.shape[0]
        else:
            end = start - 1 + step_size
        return start-1, end-1  # -1 at the end is b/c it indeces starting at 0. this is based off the shape which counts the 0 index position as a number.

    def walk_forward_window(self, split):
        walk_forward_df = self.data.copy()
        walk_forward_df = walk_forward_df[self.walk_forward_indices(split)[0]:self.walk_forward_indices(split)[1]+1]
        return walk_forward_df

    def walk_forward_dates(self, split):
        indices = self.walk_forward_indices(split)
        start_date = self.data.iloc[indices[0], ].name
        end_date = self.data.iloc[indices[1], ].name
        return start_date, end_date

    def optimization_window_dates(self, split, optimization_type='Rolling'):

        if optimization_type == 'Rolling':
            indices = self.rolling_optimization_indices(split)
            start_date = self.data.iloc[indices[0], ].name
            end_date = self.data.iloc[indices[1], ].name
        elif optimization_type == 'Standard':
            indices = self.optimization_indices(split)
            start_date = self.data.iloc[indices[0], ].name
            end_date = self.data.iloc[indices[1], ].name


        return start_date, end_date

    def split_selector(self, split, split_type):
        if split_type == 'DateRange':
            return self.date_range_split(split)
        elif split_type == 'FixedOptimizationWindow':
            return self.optimization_window(split)
        elif split_type == 'RollingOptimizationWindow':
            return self.rolling_optimization_window(split)
        elif split_type == 'WalkForwardWindow':
            return self.walk_forward_window(split)
        elif split_type == 'ExcludeDateRange':
            return self.exclude_date_range(split)
        elif split is None:
            return self.data.copy()