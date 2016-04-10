# -*- coding: utf-8 -*-

# TODO: This file need some rework to include the one tail p test the paper Zoccolan 2007 talks
# about to get correlation coefficients between activity fraction selectivity and position tolerance.


import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

import dill
import it_neuron_vrep as it
import population_utils as utils
from PositionTolerance import gaussian_position_profile as gpp

# Force reload (compile) IT cortex modules to pick changes not included in cached version.
reload(it)
reload(gpp)

if __name__ == '__main__':
    plt.ion()

    runs=500
    neurons=94
    objects=213

    # Stimulus Set
    list_of_objects = []
    for idx_n in np.arange(objects):
        list_of_objects.append('random_' + str(idx_n))

    correlation_arr = np.zeros(shape=runs)

    for r_idx in np.arange(runs):

        print ("Run %i" % r_idx)

        selectivities = np.zeros(shape=neurons)
        position_tolerances = np.zeros(shape=neurons)
        it_cortex = []

        # Create Population ------------------------------------------------------------------
        selectivities = np.random.uniform(0.1, 0.8, size=neurons)


        for n_idx in np.arange(neurons):

            # # print ("Creating neuron %i" % n_idx)
            # neuron = it.Neuron(
            #     list_of_objects,
            #     selectivity_profile='Kurtosis',
            #     position_profile='Gaussian',
            #     # size_profile='Lognormal',
            #     # rotation_profile='Gaussian',
            #     # dynamic_profile='Tamura',
            #     # occlusion_profile='TwoInputSigmoid'
            # )
            #
            # it_cortex.append(neuron)
            #
            # selectivities[n_idx] = neuron.selectivity.activity_fraction_measured
            # position_tolerances[n_idx] = neuron.position.position_tolerance

            profile = gpp.GaussianPositionProfile(selectivities[n_idx])
            position_tolerances[n_idx] =profile.position_tolerance

        plt.hist(selectivities)

        correlation_arr[r_idx] = np.corrcoef(selectivities, position_tolerances)[0, 1]
        #
        # # PLot a populations Selectivity vs position Tolerance --------------------------------
        # utils.plot_selectivity_vs_position_tolerance(it_cortex, f_size=50)
        #
        # dill.load_session('positionToleranceData.pkl')
        #
        # # Code to plot Zoccolan Data onto a figure
        # zoc_pos_tol = yScatterRaw * np.pi / 180.0  # in Radians
        # zoc_sel_af = xScatterRaw
        #
        # plt.scatter(zoc_sel_af, zoc_pos_tol, label='Original Data',  marker='o', s=60, color='blue')
        # plt.plot(
        #     xScatterRaw,
        #     (yLineFit[0]* np.pi / 180.0 * xScatterRaw + yLineFit[1] * np.pi / 180.0),
        #     color='blue', linewidth=2)
        #
        # axis = plt.gca()
        # axis.set_xlim([0,1])

#----------------------------------------------------------

    print("Average correlation between selectivity and position_tolerance %0.4f+%0.4f"
          % (np.mean(correlation_arr), np.std(correlation_arr)))
