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
around different views of an object. We model this as a single gaussian with parameters preferred
view and rotational spread around the preferred view. [1] found that tuning widths around y
(vertical) and x (horizontal) axis were similar and had an average value of 30 degrees.

Tuning curves of some neurons were found to be centered around two views. In all such cases it
was found object views of both peaks displayed some level of symmetry.A small number of neurons
also displayed view invariant responses to all object views. A sample tuning curve is shown,
however the object is not shown and it cannot be determined whether the symmetry of the object
resulted in this view invariant tuning curve. Also note that the objects used in [1] were irregular
wire or ameboid objects. Different views of the object shared some degree of symmetry but were
never completely symmetrical. Hence some of their bimodal tuning curves had amplitude differences
between the peaks. We are that if the objects were completely symmetric, peaks should have similar
amplitudes taking noise into account.

To model bimodal and invariant tuning curves, we assume each object defines a period of symmetry
around the x, y, and z axis. Within the symmetry period each object view is distinguishable by
the neuron. While an orientation outside maps to a view within the symmetry period.  Neurons are
more likely to respond to how different a view of an object looks compared to its preferred view
as opposed to absolution orientation.

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

        self.type = 'gaussian'
        self.mu = self.__get_preferred_angle()
        self.sigma = self.__get_tuning_width()

    @staticmethod
    def __get_preferred_angle():
        """
        Preferred angles are uniformly distributed over the range -pi, pi.

        :return : preferred orientation of neuron.
        """
        return np.float(np.random.uniform(low=-np.pi, high=np.pi))

    @staticmethod
    def __get_tuning_width():
        """
        Average tuning width from ref [1] also from ref [2] = 30 degrees
        Spread of tuning widths arbitrarily chosen.

        :return : rotation tuning width.
        """
        return np.float(np.random.normal(loc=(30 * np.pi / 180), scale=(50 * np.pi / 180)))

    @staticmethod
    def adjust_angles(angles, mu, period):
        """
        Return angles array such that they lie within [-period, period) of the mean. This function
        takes care of wrap effects around period edges.

        Note dimensions of angles, mu and period must be equal.

        :param  angles  : Angles to adjust
        :param  mu      : mu around width to adjust the angles
        :param period:  : Period of adjustment. For full range = 2*np.pi. For a tuning profile
                          With a rotation period of 4 =  2*np.pi / 4

        :return:        : Adjust array of angles
        """
        if not isinstance(angles, np.ndarray):
            angles = np.array([angles])
            mu = np.array([mu])
            period = np.array([period])

        min_allowed = mu - period / 2
        max_allowed = mu + period / 2

        for idx in np.arange(angles.shape[0]):

            if angles[idx] < min_allowed[idx]:
                angles[idx] = angles[idx] + period[idx]

            elif angles[idx] > max_allowed[idx]:
                angles[idx] = angles[idx] - period[idx]

        return angles

    def firing_rate_modifier(self, x, rotation_symmetry_period, mirror_symmetric):
        """

        :param  x                       : Input angles in radians
        :param  rotation_symmetry_period: Rotation symmetry period, How many times in a 360 degree
                                          rotation does the object looks like itself. Valid range
                                          {1, 360}. 1 = No rotation symmetry, 360 = compete
                                          symmetry.
        :param  mirror_symmetric         : Whether the object is mirror symmetric. Valid values
                                          = {1, 0} for each input angle.

        Note: dimensions of x, rotation_symmetry_period and mirror_symmetry mast be equal
        """
        valid_range = 2 * np.pi / rotation_symmetry_period

        # Adjust the mean to lie within the valid range
        mu_p = np.mod(self.mu, valid_range)

        # Map input angles to allowed range
        x_p = np.mod(x, valid_range)

        # Find the mirror symmetric mean, flip across the y-axis and map to the valid range
        mu_s = np.mod(-mu_p, valid_range)

        # Adjust input angles x_p such that they are defined around (-valid_range/2.valid_range/2)
        # of the target mean. This takes care of edge effects.
        x_adj = self.adjust_angles(x_p, mu_p, valid_range)
        fire_rate_p = np.exp(-(x_adj - mu_p)**2 / (2 * self.sigma**2))

        x_adj = self.adjust_angles(x_p, mu_s, valid_range)
        fire_rate_s = mirror_symmetric * np.exp(-(x_adj - mu_s) ** 2 / (2 * self.sigma ** 2))

        # Return the maximum firing rate either from the normal or mirror symmetric gaussian
        return np.maximum(fire_rate_p, fire_rate_s)

    def plot_tuning_profile(self,
                            rotation_symmetry_period=1,
                            mirror_symmetric=False,
                            axis=None,
                            font_size=20):

        if axis is None:
            f, axis = plt.subplots()

        angles = np.arange(-np.pi, np.pi, step=(1.0 / 360))
        r_symmetry_periods = np.ones_like(angles) * rotation_symmetry_period
        m_symmetries = np.ones_like(angles) * mirror_symmetric

        plt.plot(
            angles * 180 / np.pi,
            profile.firing_rate_modifier(angles, r_symmetry_periods, m_symmetries),
            linewidth=2)

        axis.set_title('Rotational Tolerance Profile', fontsize=(font_size + 10))
        axis.set_xlabel('Angle (Degrees)', fontsize=font_size)
        axis.set_ylabel('Normalized spike rate (Spikes/s)')
        axis.grid()
        axis.set_ylim([0, 1])
        axis.set_xlim([-180, 180])

        axis.tick_params(axis='x', labelsize=font_size)
        axis.tick_params(axis='y', labelsize=font_size)

        axis.annotate('Preferred Angle = %0.2f\nSpread= %0.2f'
                      % (self.mu * 180 / np.pi, self.sigma * 180 / np.pi),
                      xy=(0.95, 0.9),
                      xycoords='axes fraction',
                      fontsize=font_size,
                      horizontalalignment='right',
                      verticalalignment='top')

        axis.annotate('Rotation Symmetry Period = %0.2f\nMirror Symmetric=%s'
                      % (rotation_symmetry_period, mirror_symmetric),
                      xy=(0.95, 0.80),
                      xycoords='axes fraction',
                      fontsize=font_size,
                      horizontalalignment='right',
                      verticalalignment='top')

    def print_parameters(self):
        print("Profile            = %s" % self.type)
        print("Preferred Angle    = %0.4f (Deg)" % (self.mu * 180 / np.pi))
        print("Spread             = %0.4f (Deg)" % (self.sigma * 180 / np.pi))

if __name__ == "__main__":
    plt.ion()

    profile = GaussianRotationProfile()
    profile.print_parameters()
    profile.plot_tuning_profile()
    profile.plot_tuning_profile(mirror_symmetric=True)

    # test firing rate to an array of objects
    angle_arr = np.array([-100, 40, 30, 150])
    symmetry_periods = np.array([1, 2, 4, 360])
    mirror_symmetries = np.array([False, True, True, True])
    print profile.firing_rate_modifier(angle_arr, symmetry_periods, mirror_symmetries)

    # Test firing rate to a single object
    angle_arr = 1
    symmetry_periods = np.ones_like(angle_arr) * 1
    mirror_symmetries = np.ones_like(angle_arr) * 1

    print profile.firing_rate_modifier(angle_arr, symmetry_periods, mirror_symmetries)
