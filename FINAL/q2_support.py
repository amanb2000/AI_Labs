# This file will not be graded directly by Autolab, but you must submit it and any other files you use to answer all
# all parts of question 2
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

# Set font options (feel free to change to something you like or that works for your system)
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
matplotlib.rc("text", usetex=True)


def plot_belief_space_utility_heatmap(X, Y, Z, title='Utility', xlabel='$P_{annoyed}$', ylabel='$P_{neutral}$',
                                      n_contours=50):
    """
    Plots a utility function that is the result of value iteration as a heatmap/contour map.
    See example below for usage.
    You will use this to produce plots for varying lookahead depths (2, 5, 10) in Question 2b
    """
    font_size_contour = 14
    fig = plt.figure()
    plt.grid()
    plt.title(title, size=font_size_contour)
    filled_contour = plt.contourf(X, Y, Z, n_contours, cmap='autumn')
    plt.contour(X, Y, Z, n_contours, colors='k', linewidths=0.25)
    ax = plt.gca()
    plt.xlabel(xlabel, size=font_size_contour)
    plt.ylabel(ylabel, size=font_size_contour)
    cbar = fig.colorbar(filled_contour)
    cbar.ax.set_ylabel('Utility', size=font_size_contour)
    plt.show()

    return fig, ax


def plot_belief_space_action_boundaries(X, Y, Z, title='Action', xlabel='$P_{annoyed}$', ylabel='$P_{neutral}$',
                                        n_contours=3):
    """
    Plots a policy over a simple belief space as a heatmap/contour map.
    See example below for usage.
    You will use this to produce plots for varying lookahead depths (2, 5, 10) in Question 2b.
    """
    font_size_contour = 14
    fig = plt.figure()
    plt.grid()
    plt.title(title, size=font_size_contour)
    filled_contour = plt.contourf(X, Y, Z, n_contours, cmap='seismic')
    plt.contour(X, Y, Z, n_contours, colors='k', linewidths=0.5)
    ax = plt.gca()
    plt.xlabel(xlabel, size=font_size_contour)
    plt.ylabel(ylabel, size=font_size_contour)
    cbar = fig.colorbar(filled_contour)
    cbar.ax.set_ylabel('Action', size=font_size_contour)
    plt.show()

    return fig, ax


if __name__ == '__main__':
    # Example usage
    n = 100  # This is the resolution required in question 2b

    # Recall that p_engaged = 1. - p_annoyed - p_neutral, so we can plot functions of the belief space as 2D heatmaps/
    # contour maps.
    p_annoyed = np.linspace(0., 1., n)
    p_neutral = np.linspace(0., 1., n)
    P_annoyed, P_neutral = np.meshgrid(p_annoyed, p_neutral)  # Meshgrid creates 2D arrays for plotting

    # All invalid beliefs are zero in these plots
    utility_const = np.zeros(P_neutral.shape)
    action_simple = np.zeros(P_neutral.shape)
    for idx in range(utility_const.shape[0]):
        for jdx in range(utility_const.shape[1]):
            if P_annoyed[idx, jdx] + P_neutral[idx, jdx] <= 1.:  # This expression describes valid beliefs
                # Example utility which is just a constant (1.0) for all valid beliefs
                utility_const[idx, jdx] = 1.

                # A simple policy that maps to the action "Apologetic" (2) if the agent believes that the customer is
                # in the "Annoyed" state with probability greater than 0.5, and otherwise maps to the action "Aggressive" (0)
                action_simple[idx, jdx] = 2 if P_annoyed[idx, jdx] > 0.5 else 0

    # Display the utility and policy as contour plots (use the default values)
    plot_belief_space_utility_heatmap(P_annoyed, P_neutral, utility_const, 'Constant Utility')
    plot_belief_space_action_boundaries(P_annoyed, P_neutral, action_simple, 'Simple Policy')

    # Feel free to use this file as the starter for your solution to question 2b. It won't be used by Autolab's auto
    # grading functionality, but the code you use for 2b needs to be submitted alongside q2_support.py, e.g.:
    # tar cvf handin.tar q2_pomdp.py q2_support.py lookup_table.npy
    # where q2_support.py (this file) has been modified accordingly.

    for num in (2, 5, 10):
        path = "cache/U{}.npy".format(num)
        U = None
        with open(path, 'rb') as f:
            U = np.load(f, allow_pickle=True)
        lu = len(U)
        ls = 5
        print("\n\nLength of U: ",lu)
        A = np.zeros([lu, ls])

        for i in range(len(U)):
            A[i,:] = U[i][1]
            # print(U[i][0]," ",U[i][1])
        
        """
        Plotting Requirements
         - [x] Implement value iteration scheme. 
         - [x] Plot the *expected value* of each state.
         - [x] Plot the action taken in each state. 
         - [ ] Comment on the trend in the value function. What mistakes are made if you don't look far ahead? 
        """
        EVs = np.zeros(P_neutral.shape)
        Actions = np.zeros(P_neutral.shape)

        for idx in range(EVs.shape[0]):
            for jdx in range(EVs.shape[1]):
                if P_annoyed[idx, jdx] + P_neutral[idx, jdx] <= 1.:  # This expression describes valid beliefs
                    # Example utility which is just a constant (1.0) for all valid beliefs
                    p1 = P_annoyed[idx,jdx]
                    p2 = P_neutral[idx,jdx]
                    p3 = 1-p1-p2
                    belief = np.asarray([0,p1,p2,p3,0] )
                    EVs[idx, jdx] = np.max(A @ belief)
                    Actions[idx,jdx] = U[np.argmax(A @ belief)][0]

        # Display the utility and policy as contour plots (use the default values)
        plot_belief_space_utility_heatmap(P_annoyed, P_neutral, EVs, 'Depth {} EV'.format(num))
        plot_belief_space_action_boundaries(P_annoyed, P_neutral, Actions, 'Depth {} Actions'.format(num))
