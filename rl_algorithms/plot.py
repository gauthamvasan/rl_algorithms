import matplotlib.pyplot as plt
import numpy as np

def raw_plot(data):
    """

    Args:
        data: Numpy 2-D array. Row 0 contains episode lengths/timesteps and row 1 contains episodic returns

    Returns:

    """
    plt.plot(np.cumsum(data[0]), data[1])
    plt.xlabel('Steps')
    h = plt.ylabel("Return", labelpad=25)
    h.set_rotation(0)
    plt.pause(0.001)
    plt.show()

def smoothed_curve(returns, ep_lens, x_tick=1000, window_len=1000):
    """

    Args:
        returns: 1-D numpy array with episodic returs
        ep_lens: 1-D numpy array with episodic returs
        x_tick (int): Bin size
        window_len (int): Length of averaging window

    Returns:
        A numpy array

    """
    rets = []
    cum_episode_lengths = np.cumsum(ep_lens)

    if cum_episode_lengths[-1] >= x_tick:
        steps_show = np.arange(x_tick, cum_episode_lengths[-1] + 1, x_tick)
        for i in range(len(steps_show)):
            rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_len)) *
                                     (cum_episode_lengths < x_tick * (i + 1))]
            if rets_in_window.any():
                rets.append(np.mean(rets_in_window))

    return rets

def smoothed_plot(data):
    """

    Args:
        data: Numpy 2-D array. Row 0 contains episode lengths/timesteps and row 1 contains episodic returns

    Returns:

    """
    x_tick = 1000
    window_len = 1000

    returns = data[1]
    ep_lens = data[0]

    rets = smoothed_curve(returns=returns, ep_lens=ep_lens, x_tick=x_tick, window_len=window_len)
    plt.plot(np.arange(1, len(rets) + 1) * x_tick, rets)
    plt.xlabel('Steps')
    h = plt.ylabel("Return", labelpad=25)
    h.set_rotation(0)
    plt.pause(0.001)
    plt.show()

if __name__ == '__main__':
    data = np.loadtxt("mover0.txt"); plt.close()

    # Plot all data without smoothing
    # raw_plot(data)

    # Smoothed plot
    smoothed_plot(data)
