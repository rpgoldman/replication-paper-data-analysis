import pandas as pd
import seaborn as sns
import matplotlib as mpl
import numpy as np

from typing import Optional, Tuple, Union


def point_cloud_to_histogram(df, logscale: bool = True, channels: Tuple[str, str]=('FSC_A', 'SSC_A')):
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
    x_chan = channels[0]
    y_chan = channels[1]
    x_bins = f'{x_chan}_bins'
    y_bins = f'{y_chan}_bins'

    if logscale:
        def channel_bins(chan_name: str) -> np.ndarray:
            if chan_name in ['FSC_A', 'SSC_A']:
                bins = np.array([4 + 0.02 * x for x in range(0, 101)])
            elif chan_name in ['BL1_A']:
                bins = np.array([0.04 * x for x in range(0, 101)])
            else:
                raise ValueError(f"Don't know how to assign bins for channel {chan_name}")
            return bins

        # replace the columns for x_chan and y_chan with log values by computing the logs and
        # then renaming the log columns
        tmp = df.query(f'{x_chan} > 0 and {y_chan} > 0')
        adict = {f"{x_chan}_log": np.log10(tmp[x_chan] + 1), f"{y_chan}_log": np.log10(tmp[y_chan] + 1)}
        tmp = tmp.assign(**adict)
        tmp.drop(columns=list(channels), inplace=True)
        tmp.rename(columns={f'{x_chan}_log': x_chan, f'{y_chan}_log': y_chan}, inplace=True)
        bins4x = channel_bins(x_chan)
        bins4y = channel_bins(y_chan)
    else:
        tmp = df.copy()
        # -100_000 to 1_100_000
        bins4x = np.array([-100_000 + 10_000 * x for x in range(0, 121)])
        bins4y = bins4x
        
    tmp = tmp[list(channels)]
    tmp.loc[:, x_bins] = pd.cut(tmp[x_chan], bins=bins4x)
    tmp.loc[:, y_bins] = pd.cut(tmp[y_chan], bins=bins4y)
    tmp = tmp.dropna()
    # need an arbitrary count column
    grouped = tmp.groupby([x_bins, y_bins]).agg('count').drop(columns=[y_chan]).rename(columns={x_chan: 'events'})
    grouped.reset_index(drop=False, inplace=True)
    left_side = np.vectorize(lambda interval: interval.left)
    grouped.loc[:, x_chan] = left_side(grouped[x_bins])
    grouped.loc[:, y_chan] = left_side(grouped[y_bins])
    return grouped.drop(columns=[x_bins, y_bins]).pivot(columns=x_chan, index=y_chan).sort_index(ascending=False).droplevel(0, 'columns')

def make_heatmap(histo: pd.DataFrame, 
                 logscale: bool, 
                 ax: Optional[mpl.axes.Axes] = None, 
                ) -> mpl.axes.Axes:
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
        def ticklabels(values):
            origin = values.min()
            incr = abs(values[1] - values[0]) * 10
            my_ticklabels = [f"{x:.1f}" for x in [origin + incr * x for x in range(0, 11)]]
            return my_ticklabels
        ticklabels4x = ticklabels(histo.columns)
        ticklabels4y = ticklabels(histo.index)
        ticks = np.array([10 * x for x in range(0, 11)])
    else:
        ticklabels4x = ["%d"%(100_000 * x -100_000) for x in range(0, 13)]
        ticklabels4y = ticklabels4x.reverse()
        ticks = np.array([10 * x for x in range(0, 13)])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels4x)
    ax.set_yticklabels(ticklabels4y)
    return ax

def translate(x: Union[float, Tuple[float, float]],
              y:Optional[float] = None,
              logscale: bool = False) -> Tuple[float, float]:
    if y is None:
        assert isinstance(x, tuple) and len(x) == 2
        y = x[1]
        x = x[0]
    if logscale:
        x_origin = x.min()
        x_incr = x[1] - x[0]
        xcoord = (x - x_origin)/x_incr * 10 # origin is 4.0
        y_incr = abs(y[1] - y[0])
        y_origin = y.max() + y_incr
        ycoord = (y_origin - y)/y_incr * 10
    else:
        # x does not need origin correction
        xcoord = x + 100_000 # origin is -100_000
        # scale
        xcoord = xcoord / 10_000
        ycoord = 1_100_000 - y
        ycoord = ycoord / 10_000
    return xcoord, ycoord
