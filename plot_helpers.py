import pandas as pd
import seaborn as sns
import matplotlib as mpl
import numpy as np

from typing import Optional, Tuple, Union


def point_cloud_to_histogram(df, logscale: bool = True):
    """
    Take a Pandas DataFrame with FSC_A and SSC_A channels and transform it to a
    DataFrame of counts, effectively a histogram data structure.
    
    Parameters
    ~~~~~~~~~~
    df : DataFrame
      Pandas DataFrame with Flow Cytometry data. Must have columns for FSC_A and SSC_A.
    logscale : bool, optional
      Should we build a histogram of raw data (False) or log10-scale data (True)?
      Defaults to True.
      
    Returns
    ~~~~~~~
    DataFrame
      DataFrame with columns for FSC_A bins and rows for SSC_A bins.  The index values are
      the left (lower) side of each interval and the values are counts.
    
    Notes
    ~~~~~
    The technique of using pandas `cut` I owe to the following StackExchange
    post: https://stackoverflow.com/a/36118798/289934
    """
    
    if logscale:
        tmp = df.query('FSC_A > 0 and SSC_A > 0')
        tmp = tmp.assign(FSC_A_log=np.log10(tmp['FSC_A'] + 1), SSC_A_log=np.log10(tmp['SSC_A'] + 1))
        tmp.drop(columns=['FSC_A', 'SSC_A'], inplace=True)
        tmp.rename(columns={'FSC_A_log': 'FSC_A', 'SSC_A_log': 'SSC_A'}, inplace=True)
        bins = np.array([4 + 0.02 * x for x in range(0, 101)])
    else:
        tmp = df.copy()
        # -100_000 to 1_100_000
        bins = np.array([-100_000 + 10_000 * x for x in range(0, 121)])
        
    tmp = tmp[['FSC_A', 'SSC_A']]
    tmp.loc[:, 'FSC_A_bins'] = pd.cut(tmp['FSC_A'], bins=bins)
    tmp.loc[:, 'SSC_A_bins'] = pd.cut(tmp['SSC_A'], bins=bins)
    tmp = tmp.dropna()
    grouped = tmp.groupby(['FSC_A_bins', 'SSC_A_bins']).agg('count').drop(columns=['SSC_A']).rename(columns={'FSC_A': 'events'})
    grouped.reset_index(drop=False, inplace=True)
    left_side = np.vectorize(lambda interval: interval.left)
    grouped.loc[:, 'FSC_A'] = left_side(grouped['FSC_A_bins'])
    grouped.loc[:, 'SSC_A'] = left_side(grouped['SSC_A_bins'])
    return grouped.drop(columns=['FSC_A_bins', 'SSC_A_bins']).pivot(columns='FSC_A', index='SSC_A').sort_index(ascending=False).droplevel(0, 'columns')

def make_heatmap(histo: pd.DataFrame, logscale: bool, ax: Optional[mpl.axes.Axes] = None) -> mpl.axes.Axes:
    """
    Plot a Pandas `DataFrame`, built by `point_cloud_to_histogram`
    as a Seaborn heatmap.
    
    Parameters
    ~~~~~~~~~~
    df : DataFrame
      Pandas DataFrame with Flow Cytometry data, in histogram form.
    logscale : bool
      Should we plot raw data (False) or log-scale data?  The value must agree with the
      value of `logscale` used in generating the input data frame.
    ax : mpl.axes.Axes, optional
      If supplied, plot onto the parameter value. If not supplied, will generate a new
      Axes and return it.
      
    Returns
    ~~~~~~~
    mpl.axes.Axes
      Axes on which the heatmap was plotted.
    
    Notes
    ~~~~~
    The technique for labeling the Axes of the histogram was culled
    from StackExchange: https://stackoverflow.com/a/65970580/289934
    """
    if ax is None:
        ax = sns.heatmap(histo, cbar=True, cmap='viridis')
    else:
        sns.heatmap(histo, cbar=True, cmap='viridis', ax=ax)
    if logscale:
        ticklabels = [f"{x:.1f}" for x in [4 + 0.2 * x for x in range(0, 11)]]
        ticks = np.array([10 * x for x in range(0, 11)])
    else:
        ticklabels = ["%d"%(100_000 * x -100_000) for x in range(0, 13)]
        ticks = np.array([10 * x for x in range(0, 13)])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ticklabels.reverse()
    ax.set_yticklabels(ticklabels)
    return ax

def translate(x: Union[float, Tuple[float, float]], y:Optional[float] = None, logscale: bool = False) -> Tuple[float, float]:
    if y is None:
        assert isinstance(x, tuple) and len(x) == 2
        y = x[1]
        x = x[0]
    if logscale:
        raise NotImplementedError("Need to add heatmap translation for logscale plots")
    # x does not need origin correction
    xcoord = x + 100_000 # origin is -100_000
    # scale
    xcoord = xcoord / 10_000
    ycoord = 1_100_000 - y
    ycoord = ycoord / 10_000
    return xcoord, ycoord