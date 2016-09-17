# -*- coding: utf-8 -*-
""" ---------------------------------------------------------------------------------------------
Find the correlation between selectivity (activity fraction) and position tolerance.
Generate the plot of selectivity versus position tolerance (Figure 7 in paper).

Zoccolan Found the average correlation between these two to be -0.39+-0.01(standard error)
for their recorded population. We get a lower correlation.

Here, we find that if we use a normal distribution of AF selectivities over the range 0-0.8
as was suggested by Zoccalan, a comparable correlation is found (-0.3804+0.0833)

However if we use our IT neurons, which generate profiles based on gamma distributions, a
some what normal distribution of activity fractions is measured. Also the range of activity
fractions is smaller. Additionally, the generated correlation is also much lower
(Correlation =-0.1604 + 0.0994). But the same negative trend is still observed.

--------------------------------------------------------------------------------------------- """
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

import pickle
import it_neuron_vrep as it
from PositionTolerance import gaussian_position_profile as gpp

# Force reload (compile) IT cortex modules to pick changes not included in cached version.
reload(it)
reload(gpp)

if __name__ == '__main__':
    plt.ion()

    runs = 500
    neurons = 94
    objects = 213

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
            # print ("Creating neuron %i" % n_idx)

            # Use actual model neurons. Note that individual neurons have gamma distributed
            # object selectivities and activity fraction is calculated.
            # ------------------------------------------------------------------------------
            neuron = it.Neuron(
                list_of_objects,
                selectivity_profile='Kurtosis',
                position_profile='Gaussian',
                # size_profile='Lognormal',
                # rotation_profile='Gaussian',
                # dynamic_profile='Tamura',
                # occlusion_profile='TwoInputSigmoid'
            )
            it_cortex.append(neuron)

            selectivities[n_idx] = neuron.selectivity.activity_fraction_measured
            position_tolerances[n_idx] = neuron.position.position_tolerance

            # Alternatively just use the individual profiles directly
            # Note the top of file for an explanation of the difference in the results.
            # -------------------------------------------------------------
            # profile = gpp.GaussianPositionProfile(selectivities[n_idx])
            # position_tolerances[n_idx] = profile.position_tolerance

        correlation_arr[r_idx] = np.corrcoef(selectivities, position_tolerances)[0, 1]

        # Plot histograms of generated selectivities
        plt.figure("Selectivities Distribution")
        plt.hist(selectivities)
        plt.xlabel("Selectivity (Activity Fraction)")
        plt.ylabel("Frequency")

        if r_idx == runs - 1:
            plt.figure("Position Tolerance vs selectivity")
            plt.scatter(selectivities, position_tolerances, label="Generated Data",
                        marker='o', s=60, color='green')

            # Load the data
            with open("Zoccolan2007.pkl") as handle:
                data = pickle.load(handle)

            zoc_sel_af = data["xScatterRaw"]
            zoc_pos_tol = data["yScatterRaw"] * np.pi / 180.0

            plt.scatter(zoc_sel_af, zoc_pos_tol, label='Original Data', marker='o', s=60,
                        color='blue')

            plt.xlabel("Selectivity (Activity Fraction)")
            plt.ylabel("Position Tolerance Radians")

    print("Average correlation between selectivity and position_tolerance %0.4f+%0.4f"
          % (np.mean(correlation_arr), np.std(correlation_arr)))
