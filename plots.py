import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sampling_plot(realizations, parameters, colors, fig_width=9, fig_height=27, bins=30, alpha=0.8):
    """Creating multiple plots for the distribution of coefficient realizations.
    in one figure for a given data frame, titles and colors have to be defined
    before using the function
    Args:
        data_set (pandas.DataFrame): 
        fig_width (int, optional): Define figure width. Defaults to 20.
        fig_height (int, optional): Define figure height. Defaults to 40.
    """
    num_plots = len(realizations.columns)
    fig, ax = plt.subplots(int(np.sqrt(num_plots)+1), int(np.sqrt(num_plots)), figsize=(fig_width, fig_height))
    for k in range(int(np.sqrt(num_plots))):
        for i in range(int(np.sqrt(num_plots)+1)):
            sns.set_palette(colors)
            sns.histplot(
            data=realizations.iloc[:,(i+(k*(int(np.sqrt(num_plots)+1))))],
            bins=bins,
            alpha=alpha,
            kde=True,
            ax=ax[i,k]
            )
            ax[i,k].axhline(0, color="black", alpha=alpha)
            ax[i,k].spines["right"].set_visible(False)
            ax[i,k].spines["top"].set_visible(False)
            ax[i,k].spines["top"].set_visible(False)
            #ax[i,k].vlines(parameters)
            ax[i,k].axvline(parameters[(i+(k*(int(np.sqrt(num_plots)+1))))], c=colors[1])