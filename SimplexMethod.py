# coding=utf-8
"""
The objective function is maximization of z:
    z = 7x1 + 8x2 + 10x3

Including the following inequalities as constraints:
    2x1 + 3x2 + 1x3 <= 1000
    x1 + x2 + 2x3  <= 800

This system of equation can be converted to standard form by addition of slack (surplus in case of minimization) variables:

maximize:
    z = 7x1 + 8x2 + 10x3 + 0.s1 + 0.s2
s.t.
    2x1 + 3x2 + 1x3 + s1 + 0.s2 = 1000
    x1 + x2 + 2x3 + 0.s1 + 1.s2 = 800

The standard form can be re-written in a matrix form as follows:
[2 3 2 1 0 0 1000]
[1 1 2, 0 1 0 800]
[-7 -8 -10 0 0 1 0]

The first two rows are derived from the constraints
The last row is derived from the objective function with all elements shifted to the left hand side

"""

import numpy as np
import time


class Table:
    """
    Class for simplex computation through pivot method.
    It solves only maximization problem.
    """
    def __init__(self, array, total_slacks, total_variables):
        assert type(array) == np.ndarray, 'Please pass an array of class numpy.ndarray'
        self.array = array.copy()
        self.dims = (len(array), len(array[0]))
        self.table = array.reshape(self.dims)
        self.variable_count = total_variables
        self.slack_count = total_slacks
        self._is_optimized = False
        self.optimal_solution = None
        # Add two for constants columns and obj function value column
        assert self.variable_count + self.slack_count + 2 == self.dims[-1], \
            'The shape of the array is incorrect. Please make sure you include the correct slacks, constants and variables as columns and rows'

    def _find_pivot_column(self):
        return self.table[-1:].argmin()

    def _find_pivot_column_value(self):
        return self.table[-1:].min()

    def _find_pivot_row(self, pivot_column):
        decision_values = self.table[:, -1][:-1] / self.table[:, pivot_column][:-1]
        min_positive_value = decision_values[decision_values >= 0].min()
        # TODO: Find a better way to do this
        pivot_row = np.where(decision_values == min_positive_value)[0][0]  # The first [0] is for the tuple as where() returns a tuple;
        # Second [0] is for indexing within the array since the first element of tuple is an array containing the indices of all elements that match the case
        return pivot_row

    def _transform_pivot_row(self, pivot_row, pivot_column):
        pivot_value = self.table[pivot_row, pivot_column]
        new_pivot_row = self.table[pivot_row, :] / float(pivot_value)
        assert new_pivot_row[pivot_column] == 1
        self.table[pivot_row] = new_pivot_row

    def _transform_other_rows(self, pivot_row, pivot_column):
        assert self.table[pivot_row, pivot_column] == 1, 'Transformation did not occur.'
        pivotal_values = self.table[:, pivot_column]*-1
        for row_ind in range(self.dims[0]):
            if row_ind == pivot_row:
                continue
            else:
                self.table[row_ind] = self.table[pivot_row]*pivotal_values[row_ind] + self.table[row_ind]
                assert self.table[row_ind, pivot_column] == 0

    def _get_optimal_values(self):
        variables = []
        if self._is_optimized:
            final_row = self.table[-1]
            final_column = self.table[:, -1]
            for ind in range(self.dims[0]):
                if final_row[ind] == 0:
                    variables.append(ind)
            # assert len(variables) == len(final_column[:-1])
            self.optimal_solution = [(variables[ind], final_column[ind]) for ind in range(len(variables))]
        else:
            print('The table has not been optimized.')

    def transformation(self, iter_check=False):
        """
        Run optimization by transforming the table through simplex
        Optimal solution accessible through self.optimal_solution
        :param iter_check: if True, after each iteration is printed, user input will be required to resume
        :return: transformed table.
        """
        if self._is_optimized:
            print('Already optimized.')
            return None
        prev_pivots = None
        break_count = 0
        while self.table[-1].min() != abs(self.table[-1].min()):
            print('Looping ...')
            pivot_column = self._find_pivot_column()
            pivot_row = self._find_pivot_row(pivot_column)
            if prev_pivots:
                if prev_pivots == (pivot_row, pivot_column):
                    break_count += 1
            else:
                prev_pivots = (pivot_row, pivot_column)
            self._transform_pivot_row(pivot_row, pivot_column)
            self._transform_other_rows(pivot_row, pivot_column)
            print(self.table)
            if break_count > 3:
                print('Infinite loop. Breaking out.')
                return None
            if iter_check:
                resume_loop = input('Press Enter to continue ..')
        self._is_optimized = True
        self._get_optimal_values()
        return self.table


# Sample problem
my_array = np.array([[2, 3, 2, 1, 0, 0, 1000],
                     [1, 1, 2, 0, 1, 0, 800],
                     [-7, -8, -10, 0, 0, 1, 0]], dtype=float)

# my_array = np.array([[-1, 1, 1, 0, 0, 11],[1, 1, 0, 1, 0, 27],[2, 5, 0, 0, 1, 90],[-4, -6, 0, 0, 1, 0]], dtype=float)
# my_array = np.array([[2, 1, 1, 0, 0, 0, 18], [2, 3, 0, 1, 0, 0, 42], [3, 1, 0, 0, 1, 0, 24], [-3, -2, 0, 0, 0, 1, 0]], dtype=float)

start = time.time()
problem = Table(my_array, total_slacks=3, total_variables=2)
answer = problem.transformation(iter_check=False)
end = time.time()
time.sleep(1)
print(problem.optimal_solution)
print('Total Time Taken: {} seconds'.format(end-start))
