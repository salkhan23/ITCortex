# -*- coding: utf-8 -*-
"""
#TODO: Finish this section
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, w, b):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))


class TwoInputSigmoidOcclusionProfile:
    def __init__(self):

        self.type = 'two_input_sigmoid'
        self.d_to_t_ratio = self._get_diagnostic_group_to_total_variance_ratio()

        print self._get_d_to_t_var_ratio(3, np.sqrt(2) * 3, -0.05)

        # Todo: Choose a weight and bias
        # TODO: Check if chosen w_c and bias can generate the required ratio
        # TODO: if cannot reselect w_c and bias until can.

    @staticmethod
    def _get_diagnostic_group_to_total_variance_ratio():
        """
        Get a sample diagnostic to total ratio based on the distribution of these rations found in
        Figure 4 of Neilson - 2006. For details on the fit please see file
        diagnostic_group_variance_fit.py.(exponentially distributed with a = 6.84)

        :return: diagnostic group to total group_ratio
        """

        # Use the inverse cdf method to generate a get a sample ratio that follows the
        # distribution.
        y = np.random.uniform()

        # Inverse CDF of exponential.
        # Ref:
        # [1] http://www.ece.virginia.edu/mv/edu/prob/stat/random-number-generation.pdf
        # [2] SYSD 750 - Lecture Notes
        #
        # CDF = Fx(x) = y = 1-np.exp(-a*x)
        # ln(y-1) = -a*x, and y in uniformly distributed over 0, 1
        x = -np.log(1 - y) / 6.84

        return x

    @staticmethod
    def _get_combined_weight_and_bias():
        """
        Return a w_combined and bias pair that can generate the specified
        diagnostic group to total group_ratio
        :return:
        """
        pass

    # noinspection PyTypeChecker
    @staticmethod
    def _get_d_to_t_var_ratio(w_c, w_d, b):
        """
        Given a weight for the combined axis and the diagnostic axis, find the
        diagnostic group to total variance ratio.

        :param w_c: weight_combined
        :param w_d: weight_diagnostic
        :param b: bias
        :return: diagnostic group to total variance ratio
        """
        w_n = (np.sqrt(2) * w_c) - w_d

        vis_arr = np.arange(1, step=0.05)
        vis_arr = np.reshape(vis_arr, (vis_arr.shape[0], 1))

        rates_n = sigmoid(vis_arr, w_n, b)
        rates_d = sigmoid(vis_arr, w_d, b)

        mean_n = np.mean(rates_n)
        mean_d = np.mean(rates_d)

        mean_overall = np.mean(np.append(rates_n, rates_d))
        var_overall = np.var(np.append(rates_n, rates_d))

        var_diagnostic_group = ((mean_n - mean_overall) ** 2 + (mean_d - mean_overall) ** 2) / 2

        ratio = var_diagnostic_group / var_overall

        return ratio


if __name__ == "__main__":
    plt.ion()

    profile = TwoInputSigmoidOcclusionProfile()
