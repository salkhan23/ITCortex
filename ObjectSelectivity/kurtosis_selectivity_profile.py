# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

from power_law_selectivity_profile import calculate_activity_fraction

__author__ = 'bptripp'


def calculate_kurtosis(rates_per_object):
    """
    Given an array of firing rates of the neuron to objects, return the sparseness metric
    Kurtosis (actually excess kurtosis) of the neuron as defined in:

    [1]  Lehky, S. R., Kiani, R., Esteky, H., & Tanaka, K. (2011). Statistics of
         visual responses in primate inferotemporal cortex to object stimuli.
         Journal of Neurophysiology, 106(3), 1097–117.


    Kurtosis  =  (sum (Ri - Rmean)**4 / (n*sigma**4)) - 3

    :param rates_per_object: array of firing rates of the neuron to multiple objects.
    :return: kurtosis sparseness.

    This is defined outside the class as it is used by other selectivity profiles.
    """
    n = np.float(rates_per_object.shape[0])

    rates_mean = np.mean(rates_per_object)
    rates_sigma = np.std(rates_per_object)

    kurtosis = np.sum((rates_per_object - rates_mean)**4) / (n * rates_sigma**4) - 3

    # kurtosis2= np.sum((rates_per_object - rates_mean)**4) / n \
    #            / (np.sum((rates_per_object - rates_mean)**2) / n)** 2 - 3

    return kurtosis


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

        obj_preferences = gamma.rvs(self.a, loc=0, scale=self.b, size=len(list_of_objects))
        obj_preferences = obj_preferences / self.get_max_firing_rate()

        self.objects = {item: obj_preferences[item_idx]
                        for item_idx, item in enumerate(list_of_objects)}

        # TODO: Remove this code and function, above is a faster way to generate object preferences
        # self.objects = {item: self.__get_object_preference(np.float(np.random.uniform(size=1)))
        #                 for item in list_of_objects}

        self.activity_fraction_measured = \
            calculate_activity_fraction(np.array(self.objects.values()))

        # To calculate absolute activity fraction, the stimuli set consists of all objects the
        # neuron responds. Model this by getting firing rates distributed over the entire cdf
        #  with a small step
        rates_distribution_for_cdf = np.linspace(start=0, stop=1, num=1000, endpoint=False)
        rates_all_obj = self.__get_object_preference(rates_distribution_for_cdf)

        self.activity_fraction_absolute = \
            calculate_activity_fraction(rates_all_obj)

        # Calculate the excess kurtosis of the neuron
        self.kurtosis_absolute = 6.0 / self.a
        self.kurtosis_measured = calculate_kurtosis(np.array(self.objects.values()))

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
        return np.float(gamma.rvs(37.4292, scale=0.062, loc=0, size=1))

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
        print("Sparseness(absolute kurtosis)             = %0.4f" % self.kurtosis_absolute)
        print("Sparseness(measured kurtosis)             = %0.4f" % self.kurtosis_measured)
        print("Object Preferences                        = ")

        max_name_length = np.max([len(name) for name in self.objects.keys()])

        lst = self.get_ranked_object_list()
        for obj, rate in lst:
            print ("\t%s : %0.4f" % (obj.ljust(max_name_length), rate))

    def plot_object_preferences(self, axis=None, font_size=34, print_parameters=True):
        """ Plot Neurons Object Preferences
        :param axis: axis to plot in. [default=None]
        """
        lst = self.get_ranked_object_list()
        objects, rate = zip(*lst)
        x = np.arange(len(rate))

        if axis is None:
            fig_obj_pref, axis = plt.subplots()

        axis.plot(x, rate, marker='o', markersize=15, linewidth=2)

        # axis.set_title("Object Selectivity", fontsize=font_size)
        axis.set_xticklabels(objects, size='small')
        axis.set_xlabel('Ranked Objects', fontsize=font_size)
        axis.set_ylabel('FR (Spikes/s)', fontsize=font_size)
        axis.grid()

        axis.set_ylim([0, 1])

        max_fire = self.get_max_firing_rate()

        if print_parameters:
            axis.annotate(r'$SI_K=%0.2f$' % (self.kurtosis_measured),
                          xy=(0.95, 0.9),
                          xycoords='axes fraction',
                          fontsize=font_size,
                          horizontalalignment='right',
                          verticalalignment='top')

            axis.annotate(r'$SI_{AF}=%0.2f$' % self.activity_fraction_measured,
                          xy=(0.95, 0.75),
                          xycoords='axes fraction',
                          fontsize=font_size,
                          horizontalalignment='right',
                          verticalalignment='top')

        axis.tick_params(axis='x', labelsize=font_size)
        axis.tick_params(axis='y', labelsize=font_size)


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

    for ii in np.arange(len(obj_list), 806):
        obj_list.append('random_' + str(ii))

    profile1 = KurtosisSparseness(obj_list)
    profile1.get_ranked_object_list()
    # profile1.print_parameters()

    print("Sparseness(absolute kurtosis)             = %0.4f" % profile1.kurtosis_absolute)
    print("Sparseness(measured kurtosis)             = %0.4f" % profile1.kurtosis_measured)


    selectivities_abs = []
    selectivities_meas = []
    mean_rates = []

    for idx in np.arange(674):
         print idx
         profile = KurtosisSparseness(obj_list)

         mean_rates.append(np.mean(profile.objects.values())* profile.get_max_firing_rate())
         selectivities_meas.append(profile.kurtosis_measured)

    mean_rates = np.array(mean_rates)
    selectivities_abs = np.array(selectivities_abs)
    selectivities_meas = np.array(selectivities_meas)


    f, axis = plt.subplots()
    font_size=24
    axis.scatter(mean_rates, np.log(selectivities_meas + 1), color='g', s=60)
    # axis.loglog(mean_rates, selectivities_meas + 1, 'go', fontsize=font_size)
    axis.set_xlabel("log(Average Fire Rate) (spikes/second)", fontsize=font_size)
    axis.set_ylabel("log(Object Selectivity + 1)", fontsize=font_size)
    axis.grid()
    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)

    plt.figure()
    plt.scatter(mean_rates, selectivities_meas, color='g', s=60)




    # print ("Max Firing Rate: %0.4f" % profile1.get_max_firing_rate())
