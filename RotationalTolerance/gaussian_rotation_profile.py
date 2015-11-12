# -*- coding: utf-8 -*-
"""

Model rotation tolerance as a single gaussian rv, with mu = preferred angle and sigma = the spread
of the tolerance.

Ref:
[1] Logothesis, Pauls, Poggio - 1995 -
    Shape representation in the inferior temporal cortex of monkeys.

[2] Hung, Carlson, Conner - 2012 -
    Medial Axis Shape Coding in Macaque Inferotemporal Cortex.

IT neurons typically had a preferred view of an object - most familiar view - and their responses
gradually declined as views shifted away (bell shaped function). Individual neurons are tuned
around different views of an object. Average tuning curve width was 29 degrees. Fit with a single
gaussian. Tuning widths around y (vertical) and x (horizontal) axis were similar.

Tuning curves of some neurons were found to be centered around two views. In all such cases it
was found object views of both peaks displayed some level of symmetry. A small number of neurons
also displayed view invariant responses to all object views. A sample tuning curve is shown,
however the object is not shown and it cannot be determined whether the symmetry of the object
resulted in this view invariant tuning curve.

For objects that display some symmetry in their rotation tuning, we defined symmetry periods
around the x, y and z axes. Within the symmetry period each view is distinguishable by the neuron,
while an orientation outside maps to a view within the symmetry period. Neurons are more likely to
respond to how different a view of an object looks compared to its preferred view as opposed to
its absolution orientation.

For example a mirror symmetry objects suc as cars/faces will have a y-axis symmetry period of
180 degrees (mirror symmetry). An arbitrary preferred view of the object will share some symmetry
with the view of the neuron at preferred view + 180. Similarly a ball will have a symmetry period
of 0 - completely symmetric - around any axis. Regardless of the actual orientation of the ball it
will always look a neurons preferred view, from a purely rotation tuning perspective. This does not
imply symmetry objects will generate larger responses, as many factors are involved in the
calculation of the net firing rate of the neuron, such as object preference, location etc...

Each neuron than has a single rotation tuning curve that ranges from -180, 180 for all objects it
is selective for. However if an individual object defines a period of symmetry it will only see a
partial view of the complete tuning profile within the defined period of symmetry. Currently we
assume symmetric tuning of orientations away from the preferred view in both directions.

The rotational tolerance around the x-axis and y-axis were similar and stayed view dependent.
Similar tuning widths. Similar results were seen in [2]. Measure of tuning width is different but
the results equate to the same - [1] using sigma of the gaussian fit, [2] uses the  range of
degrees over with the response remains significantly more than background.

Rotation tuning around the z -axis, picture place rotation, were found to be initially broader
than tuning widths around x and y and over the coarse of experiments, consistently broadened to
become view invariant. Similar results were seen in ref 2. This is not modeled.

@author: s362khan
"""
import numpy as np
import matplotlib.pyplot as plt


class GaussianRotationProfile:

    def __init__(self):

        self.type = 'Gaussian'
        self.mu = self.__get_preferred_angle()
        self.sigma = self.__get_tuning_width()

    @staticmethod
    def __get_preferred_angle():
        """
        Preferred angles are uniformly distributed over the range -pi, pi.

        :return : preferred orientation of Neuron.
        """
        return np.float(np.random.uniform(low=-np.pi, high=np.pi))

    @staticmethod
    def __get_tuning_width():
        """
        Average tuning width from ref [1], also from ref [2].
        Spread of tuning widths arbitrarily chosen.

        :return : rotation tuning width.
        """
        return np.float(np.random.normal(loc=30 * np.pi / 180, scale=50 * np.pi / 180))

    def __adjust_angles(self, angles, mu):
        '''Return angles such that they lie within [-pi, pi) of the mean, mu '''

        if not isinstance(angles, (np.ndarray)):
            angles = np.array(angles)

        angles = [angle + 2*np.pi if angle < (mu - np.pi) else angle for angle in angles]
        angles = [angle - 2*np.pi if angle > (mu + np.pi) else angle for angle in angles]


        return np.array(angles)

    def firing_rate_modifier(self, angles):

        angles = self.__adjust_angles(angles, self.mu)

        return(np.exp(-(angles - self.mu)**2 / (2.0*self.sigma**2)))


    def plot_profile(self, angles=np.linspace(-np.pi, np.pi, num=200), axis = None):

        if axis is None:
            f, axis = plt.subplots()

        axis.plot(angles, self.firing_rate_modifier(angles))

        axis.set_title('Rotational Tolerance Profile')
        axis.set_xlabel('angle (Radians)')
        axis.set_ylabel('Normalized Spike Rate (Spikes/s)')
        axis.grid()
        axis.set_ylim([0, 1])
        axis.set_xlim(min())
        axis.legend(loc=1, fontsize='small')

if __name__ == "__main__":
    plt.ion()

    profiles = GaussianRotationProfile()
    profiles.plot_profile()
