import matplotlib.pyplot as plt
import seaborn as sns

def sampling_plot(realizations, colors, fig_width=9, fig_height=27, bins=30, alpha=0.8):
    """Creating multiple plots for the distribution of coefficient realizations.
    in one figure for a given data frame, titles and colors have to be defined
    before using the function
    Args:
        data_set (pandas.DataFrame): 
        fig_width (int, optional): Define figure width. Defaults to 20.
        fig_height (int, optional): Define figure height. Defaults to 40.
    """
    num_plots = len(realizations.columns)
    fig, ax = plt.subplots(num_plots, 1, figsize=(fig_width, fig_height))

    for i in range(num_plots):
        sns.set_palette(colors)
        sns.histplot(
            data=realizations.iloc[:,i],
            bins=bins,
            alpha=alpha,
            kde=True,
            ax=ax[i]
        )
        ax[i].axhline(0, color="black", alpha=alpha)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)