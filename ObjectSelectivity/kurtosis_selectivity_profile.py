# -*- coding: utf-8 -*-
__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

from power_law_selectivity_profile import get_activity_fraction


class KurtosisSparseness:

    def __init__(self, list_of_objects):
        """
        A statistical model of selectivity & max spike rate distribution based on:

        Lehky, S. R., Kiani, R., Esteky, H., & Tanaka, K. (2011). Statistics of
            visual responses in primate inferotemporal cortex to object stimuli.
            Journal of Neurophysiology, 106(3), 1097–117.

        This paper shows (with a large passive-viewing IT dataset) that selectivity
        (kurtosis of neuron responses to different images) is lower than sparseness
        (kurtosis of population responses to each image). They propose a simple
        model (see pg. 1112) that explains this difference in terms of heterogeneity
        of spike rate distributions.

        They model these spike rate distributions as gamma functions,  We don't directly use
        Lehky et al.'s distribution of gamma PDF parameters, since probably some of their
        variability was due to stimulus parameters such as size, position, etc. being
        non-optimal for many neurons. We do use their shape-parameter distribution, but we
        use a different scale parameter distribution that approximates that of Lehky et al.
        after being scaled by a realistic distribution of scale factors for non-optimal size,
        position, etc. For derivation of scale factors see kurtosis_fit.py

        Additionally a function is provided to get the max firing rate of the neuron. Once
        parameters of the gamma distribution over the objects is calculated, we take the point at
        which the CDF = 0.99 as the maximum.
        """
        self.type = 'kurtosis'

        self.a = self.__get_distribution_shape_parameter()
        self.b = self.__get_distribution_scale_parameter()

        self.objects = {item: self.__get_object_preference(np.random.uniform(size=1))
                        for item in list_of_objects}

        self.activity_fraction_measured = \
            get_activity_fraction(np.array(self.objects.values()))

        # To calculate absolute activity fraction, the stimuli set consists of all objects the
        # neuron responds. Model this by getting firing rates distributed over the entire cdf
        #  with a small step
        rates_distribution_for_cdf = np.linspace(start=0, stop=1, num=100, endpoint=False)
        rates_all_obj = self.__get_object_preference(rates_distribution_for_cdf)

        self.activity_fraction_absolute = \
            get_activity_fraction(rates_all_obj)

        # Calculate the excess kurtosis of the neuron
        self.kurtosis = 6.0 / self.a

    @staticmethod
    def __get_distribution_shape_parameter():
        """
        Get sample shape parameter for the gamma distribution of firing rates over objects.
        See derivation in kurtosis_fit.py.

        :rtype : shape parameter.
        """
        shape_param = np.float(gamma.rvs(4.0, scale=0.5, loc=0, size=1))

        return np.maximum(1.01, shape_param)  # Avoid making PDF go to infinity at zero spike rate.

    @staticmethod
    def __get_distribution_scale_parameter():
        """
        Get sample shape parameter for the gamma distribution of firing rate over objects.
        See derivation in kurtosis_fit.py.

        :rtype : scale parameter.
        """
        return np.float(gamma.rvs(4.6987, scale=0.3200, loc=0, size=1))

    def __get_object_preference(self, cdf_loc):
        """
        Use the inverse cdf to get a random firing rate modifier, its normalized firing rate.

        :rtype : Firing rate modifier
        """
        obj_pref = gamma.ppf(cdf_loc, self.a, scale=self.b, loc=0)

        return obj_pref / self.get_max_firing_rate()

    def get_max_firing_rate(self):
        """
        Return the maximum firing rate of the neuron. Where the maximum firing rate is defined
        as the rate at which the CDF =0.99

        :return: maximum firing rate of the neuron.
        """
        return gamma.ppf(0.99, self.a, scale=self.b, loc=0)

    def get_ranked_object_list(self):
        """ Return neurons rank list of objects and rate modification factors """
        return sorted(self.objects.items(), key=lambda item: item[1], reverse=True)

    def print_parameters(self):
        """ Print parameters of the selectivity profile """
        print("Profile                                   = %s" % self.type)
        print("Sparseness(absolute activity fraction)    = %0.4f"
              % self.activity_fraction_absolute)
        print("Sparseness(measured activity fraction)    = %0.4f"
              % self.activity_fraction_measured)
        print("Sparseness (kurtosis)                     = %0.4f" % self.kurtosis)
        print("Object Preferences                        = ")

        max_name_length = np.max([len(name) for name in self.objects.keys()])

        lst = self.get_ranked_object_list()
        for obj, rate in lst:
            print ("\t%s : %0.4f" % (obj.ljust(max_name_length), rate))


if __name__ == '__main__':
    plt.ion()

    obj_list = ['car',
                'van',
                'Truck',
                'bus',
                'pedestrian',
                'cyclist',
                'tram',
                'person sitting']

    profile1 = KurtosisSparseness(obj_list)
    profile1.get_ranked_object_list()
    profile1.print_parameters()
    print ("Max Firing Rate: %0.4f" % profile1.get_max_firing_rate())
