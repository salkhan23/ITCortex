import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.optimize as so


def integrate(dt, a, b, c, x, u):
    """
    Euler integration of state-space equations.

    :param dt: Time step (s; should be ~5ms)
    :param a: The standard feedback matrix from linear-systems theory
    :param b: Input matrix
    :param c: Output matrix
    :param x: State vector
    :param u: Input vector
    :return: (x, y), i.e. the state and the output
    """
    dxdt = np.dot(a, x) + np.dot(b, u)
    x = x + dxdt * dt

    y = np.dot(c, x)

    return x, y


def dynamics_profile(
        x_arr, early_tau, late_tau, early_gain, late_gain, latency):
    """

    :param latency:
    :param x_arr:
    :param early_tau:
    :param late_tau:
    :param early_gain:
    :param late_gain:
    :param late_sustained_gain:
    :return:
    """
    # print early_tau, late_tau, early_gain, late_gain, latency

    dt = 0.005
    avg_fire_rate = 51.3

    if not isinstance(x_arr, (list, np.ndarray)):
        x_arr = np.array([x_arr])

    # Create a time array large enough to get the dynamic rate for the input time x
    time_arr = np.arange(0, max(x_arr) + dt, step=dt)

    # we assume tha average firing rate is fed in for 400ms
    input_arr = [avg_fire_rate if t < 0.4 else 0 for t in time_arr]

    # Create the dynamic profile ---------------------------------------------------------------
    early_A = np.array([[-1, 0], [1, -1]])
    early_B = np.array([1, 0])

    late_A = np.array([[-1, 0], [1, -1]])
    late_B = np.array([1, 0])

    early_C = np.array([early_gain, -early_gain])
    late_C = np.array([late_gain, - late_gain])

    early_x = np.zeros((2, 1))
    late_x = np.zeros((2, 1))

    # Assume there is no latency involved
    dynamics_rates = np.zeros_like(input_arr)
    early_dynamics = np.zeros_like(input_arr)
    late_dynamics = np.zeros_like(input_arr)

    for s_idx, static_fr in enumerate(input_arr):

        # In the actual model, we assume early selectivity is twice the regular selectivity.
        # But it peaks at 1. For now just assume the rate is doubled. This is the same for
        # the original model most of the time.
        early_rate = 2 * input_arr[s_idx]

        if s_idx < latency:
            late_rate = 0
        else:
            late_rate = input_arr[s_idx - int(latency)]

        # Step Dynamics
        early_a = 1 / early_tau * early_A
        early_b = 1 / early_tau * early_B

        early_x[:, 0], early_y = \
            integrate(dt, early_a, early_b, early_C, early_x[:, 0], early_rate)

        late_a = 1 / late_tau * late_A
        late_b = 1 / late_tau * late_B

        late_x[:, 0], late_y = \
            integrate(dt, late_a, late_b, late_C, late_x[:, 0], late_rate)

        dynamics_rates[s_idx] = np.maximum(0, early_y) + np.maximum(0, late_y)
        early_dynamics[s_idx] = early_y
        late_dynamics[s_idx] = late_y

        # print s_idx, ' : ', 'dynamics_rate:', dynamics_rates[s_idx], 'late_x', late_x.T,
        # "input", late_rate

    # plt.plot(time_arr, early_dynamics, label='Early LTI system')
    # plt.plot(time_arr, late_dynamics, label='Late LTI system')

    # Return the dynamic rate for each x(time) value
    out_arr = np.zeros_like(x_arr)
    for x_idx, x in enumerate(x_arr):

        index = np.rint(x / dt)
        out_arr[x_idx] = dynamics_rates[index]

    return out_arr


if __name__ == '__main__':
    plt.ion()
    # plt.figure()

    with open("tamura2001Updated.pkl", 'rb') as fid:
        list_of_neurons = pickle.load(fid)

    neuron = list_of_neurons[1]
    # Most preferred object for neuron in fig 3 at list_of_neurons[4]
    # most preferred object for neuron in fig 4 at list_of_neurons[1]
    # TODO: don't forget to update the average firing rate  in dynamics_profile function

    # Fix a small typo in retrieved dictionary
    list_of_neurons[6]["avg_rate"] = list_of_neurons[6]["avg rate"]

    for n_idx, n in enumerate(list_of_neurons):
        print n_idx, n["avg_rate"]

    # Find the first index where the rate is greater than 20
    cutoff_idx = 0
    for r_idx, rate in enumerate(neuron['rate']):
        if rate > 20:
            cutoff_idx = r_idx
            break

    meas_rate = neuron["rate"][cutoff_idx:]
    meas_time = neuron["time"][cutoff_idx:]

    # Adjust the time array. This model assumes no latencies
    meas_time = meas_time - neuron["time"][cutoff_idx]
    meas_time = meas_time / 1000.0  # change to seconds

    plt.scatter(meas_time, meas_rate)

    p0 = [0.05, 0.25, 4.38, 2, 27]
    coeffs, matcov = so.curve_fit(dynamics_profile, meas_time, meas_rate, p0=p0)
    print coeffs
    print "standard error of fit:"
    print np.sqrt(np.diag(matcov))

    # Create a profile with the best fit params
    # coeffs = [0.02, 0.09, 4.38, 2.72, 0,20]
    t_arr = np.arange(0, 0.8, 0.005)
    out_rates = dynamics_profile(
        t_arr, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4])

    plt.plot(
        t_arr,
        out_rates,
        label=r'$\tau_{early}=%0.2f,\ \tau_{late}=%0.2f,\ g_{early}=%0.2f,\ g_{late}=%0.2f,'
              r'\ addLatency=%0.2f$'
              % (coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4])
    )

    plt.legend()

    plt.title("Neuron %d in Figure %d" % (neuron['neuron_idx'], neuron['fig_idx']))

    # plt.plot(t_arr, dynamics_profile(t_arr, 0.0207,0.0486,4.14,3.58,0.49886,30))
