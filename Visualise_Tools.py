import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


from sklearn.manifold import TSNE
class Histogram:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train


    def plot(self):
        plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

        # Plot Histogram on x
        plt.hist(self.x_train[:,6], bins=30)
        plt.gca().set(title='Frequency Histogram', ylabel='Frequency')

        plt.show()
            
        plt.hist(self.y_train, bins=10)
        plt.gca().set(title='Frequency Histogram', ylabel='Frequency')


class T_SNE:
    def __init__(self, x_train, y_train):
            self.x_train = x_train
            self.y_train = y_train


    def train(self):
        tsne = TSNE(n_components=3, verbose=1, random_state=123, perplexity=40)
        self.z = tsne.fit_transform(self.x_train) 
        bins = np.linspace(0, 100000,5)
        digitized = np.digitize(np.linalg.norm(self.y_train, axis=1), bins)*20000
        

        self.df = pd.DataFrame()
        self.df["y"] = digitized
        self.df["comp-1"] = self.z[:,0]
        self.df["comp-2"] = self.z[:,1]
        self.df["comp-3"] = self.z[:,2]

        self.n = self.df["y"].nunique()
    def plot(self): 

        sns.scatterplot(x="comp-2", y="comp-3", hue=self.df.y.tolist(),
                        palette=sns.color_palette("hls", self.n),
                        data=self.df).set(title="Iris data T-SNE projection")

class CorrelationPlot:
    def __init__(self, features):
        self.features = features


    def plot(self):

        plt.figure(figsize=(30, 15))
        sns.set(font_scale=1.2)
        # Store heatmap object in a variable to easily access it when you want to include more features (such as title).
        # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
        heatmap = sns.heatmap(self.features.corr(), vmin=-1, vmax=1, annot=True)
        heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 20)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 20, ha='left')
        # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
        # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=16)
        plt.yticks(rotation=0) 

        plt.draw()  # this is needed because get_window_extent needs a renderer to work
        yax = heatmap.get_yaxis()
        # find the maximum width of the label on the major ticks
        pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
        yax.set_tick_params(pad=pad)
        #------------------------------------------------------------------------   
        plt.figure(figsize=(24, 12))
        ax = plt.subplot(1,3,1)
        heatmap = sns.heatmap(self.features.corr()[['$\Delta V_{1x}$']].sort_values(by='$\Delta V_{1x}$', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
        heatmap.set_title('Features Correlating with $V1_{x}$', fontdict={'fontsize':18}, pad=16)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(),ha='left')
        plt.yticks(rotation=0)

        plt.draw()  # this is needed because get_window_extent needs a renderer to work
        yax = heatmap.get_yaxis()
        # find the maximum width of the label on the major ticks
        pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
        yax.set_tick_params(pad=pad)
        #------------------------------------------------------------------------
        plt.subplot(1,3,2)
        heatmap = sns.heatmap(self.features.corr()[['$\Delta V_{1y}$']].sort_values(by='$\Delta V_{1y}$', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
        heatmap.set_title('Features Correlating with $\Delta V_{1y}$', fontdict={'fontsize':18}, pad=16)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), ha='left')
        plt.yticks(rotation=0)

        plt.draw()  # this is needed because get_window_extent needs a renderer to work
        yax = heatmap.get_yaxis()
        # find the maximum width of the label on the major ticks
        pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
        yax.set_tick_params(pad=pad)
        #------------------------------------------------------------------------
        plt.subplot(1,3,3)
        heatmap = sns.heatmap(self.features.corr()[['$\Delta V_{1z}$']].sort_values(by='$\Delta V_{1z}$', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
        heatmap.set_title('Features Correlating with $\Delta V_{1z}$', fontdict={'fontsize':18}, pad=16)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), ha='left')
        plt.yticks(rotation=0) 

        plt.draw()  # this is needed because get_window_extent needs a renderer to work
        yax = heatmap.get_yaxis()
        # find the maximum width of the label on the major ticks
        pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
        yax.set_tick_params(pad=pad)

        plt.figure(figsize=(24,12))
        heatmap = plt.plot(self.features.corr()[['$\Delta V_{1x}$']].sort_values(by='$\Delta V_{1x}$', ascending=True))
        heatmap.title('Features Correlating with $\Delta V_{1x}$', fontdict={'fontsize':18}, pad=16)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 20, ha='left')
        plt.yticks(rotation=0) 

        sns.pairplot(self.features)


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def format_axes(ax, fontsize, xlabel, ylabel, scale_legend=False, force_ticks=None):
    ax.legend(fontsize = fontsize)

    ax.xaxis.offsetText.set_fontsize(fontsize)
    ax.yaxis.offsetText.set_fontsize(fontsize)

    plt.xticks(fontsize= fontsize)
    if np.all(force_ticks):
        ax.set_xticks(force_ticks)
        plt.minorticks_off()
        if all(force_ticks<1e-3):
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
    plt.yticks(fontsize= fontsize)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    markerscale = 1
    # ax.tick_params(labelsize=40)
    if scale_legend:
        markerscale *= 5
    lgnd = ax.legend(fontsize=fontsize, markerscale=markerscale)
