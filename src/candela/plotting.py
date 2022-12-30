"""This submodule includes functions for plotting and visualisation of anomalies.

The submodule exposes two functions: `plot_features` and `show_trajectories2D`. The first is dedicated to plotting graphs with one or more features and the option of overlaying detected anomalies. The second is for the representation of spatial data in 2D, with a possible color coded third dimension, and again with the possible overlay of anomalies.

The functions make easy the integration with matplotlib (for as much the integration of anything with matplotlib can be easy). Combining efforts with the other two submodules of candela it is possible to obtain a complex two panles figure in four lines writing:

```python
import pandas as pd
from candela.functions import local_outlier_factor
from candela.particles2d import preprocess
from candela.plotting import plot_features, show_trajectories2D

df = pd.read_csv("my_trace.csv")
trace = preprocess(df, track_id=0)

outliers_velocity_LOF = local_outlier_factor(trace.vtot)

fig, axs = plt.subplots(1, 2, figsize = (12,7))
plot_features('frame_id','vtot',outliers=outliers_velocity_LOF, outlierLabels="Anomalies", data=track, showPoints=True, ax=axs[0])
show_trajectories2D('x','y','vtot',outliers=outliers_velocity_LOF, data=track, outlier_names="Velocity anomaly", cbarName = "Velocity (km/h)")
plt.show()
```
"""
from typing import Sequence, Union, Mapping, Tuple, Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker
import warnings
from itertools import chain
from matplotlib.colors import to_rgb

styles = [[65, "black", "d"], [100, "red", "*"],
          [90, "blue", "3"], [90, "green", "2"]]

__pdoc__ = {}
__pdoc__["__2Dplot"] = True
__pdoc__["__vec_from_index"] = True
__pdoc__["__index_from_vec"] = True
__pdoc__["__to_sequence"] = True
__pdoc__["__check_columns"] = True
__pdoc__["__edges"] = True
__pdoc__["__determine_handles_labels"] = True
__pdoc__["__add_outliers"] = True
__pdoc__["__draw_features"] = True


def plot_features(x: Union[Sequence, str], y: Union[Sequence[Sequence], Sequence[str]], outliers: Union[str, Sequence[str], Sequence[int], Sequence[bool], None] = None, featureLabels: Union[str, Sequence[str], None] = None, featureKW: Sequence[Mapping[str, any]] = [], outlierLabels: Union[str, Sequence[str], None] = None, useOutlierLabelArray: Union[bool, str] = "auto", showPoints: bool = False, data: Union[pd.DataFrame, None] = None, ax: Union[plt.Axes, None] = None) -> Tuple[Figure, Axes]:
    """High level function for plotting features, possibly with detected anomalies.

    It can be used in different ways depending on the input:

    1. it can plot features alone allowing for some fine tuning of the curve appearance;
    2. it can highlight a single set of anomalies on all features, with or without showing the fist and last point of every anomalous tract. In this case it also shows a point for every isolated anomaly that would be otherwise hidden.
    3. it can show a different set of anomalies for every feature.
    4. it can show a single feature with overlaid different sets of anomalies.


    Args:
        x (Union[Sequence, str]): The values of x, these are the same for all the features, can be the name of a column in data.
        y (Union[Sequence[Sequence], Sequence[str]]): A sequence of sequences of values for different features. The inner sequences can be replaced with the names of columns in data.
        outliers (Union[str, Sequence[str], Sequence[int], Sequence[bool], None], optional): Sequences of detected anomalies to be plotted on the features. The inner sequences can be replaced with the names of columns in data. Defaults to None.
        featureLabels (Union[str, Sequence[str], None], optional): Label or list of labels for the features. If provided should be as many as the features, overrides the column names. Defaults to None.
        featureKW (Sequence[Mapping[str, any]], optional): Dictionaries of keywords to adjust the properties of every feature curve. If provided should contain as many dictionaries (even empty) as the features. Defaults to [].
        outlierLabels (Union[str, Sequence[str], None], optional): Labels to represent the anomalies in the legend. If provided should be as many as the oultiers, overrides the column names if passed. Defaults to None.
        useOutlierLabelArray (Union[bool, str], optional): If `True` teat the vectors containing the anomalies info as arrays of labels. Otherwise the vectors are supposed to contain the indexes of the anomalies. The keyword "auto" activates a simple heuristics to determine the correct handling. Defaults to "auto".
        showPoints (bool, optional): Shows fist and last point of every anomalous tract. It also shows a point for every isolated anomaly that would be otherwise hidden. Defaults to False.
        data (Union[pd.DataFrame, None], optional): An optional dataframe containing data. If passed it's possible to specify x, y and outliers with strings referring to columns in `data`. Defaults to None.
        ax (Union[plt.Axes, None], optional): An instance of `matplotlib.axes.Axes` used to plot the graph. Defaults to None.

    Returns:
        Tuple[Figure, Axes]: Matplotlib figure and axes containing the plot.
    """
    # adjusting format
    y = __to_sequence(y)
    featureLabels = __to_sequence(featureLabels)
    featureKW = __to_sequence(featureKW)
    outliers = __to_sequence(outliers)
    outlierLabels = __to_sequence(outlierLabels)

    # checking lengths and data presence
    assert len(featureLabels) in [0, len(
        y)], "If selecting custom labels for the features, you should pass as many as the features."
    assert len(featureKW) in [0, len(y)]
    assert len(outlierLabels) in [0, len(outliers)]
    assert len(outliers) in [0, 1, len(y)] or len(y) == 1

    column_data = [dat for dat in chain(y, outliers) if isinstance(dat, str)]
    if isinstance(x, str):
        column_data.append(x)
    assert len(column_data) == 0 or data is not None
    __check_columns(data, column_data)

    # getting x
    if isinstance(x, str):
        x = [data[x], x]
    else:
        x = [x, "x"]

    # getting y
    features = []
    for i, feat in enumerate(y):
        if isinstance(feat, str):
            t_feat = data[feat]
            lab = feat
        else:
            t_feat = feat
            lab = f"Feature {i}"
        if featureLabels:
            lab = featureLabels[i]
        featKW = featureKW[i] if featureKW else {}
        features.append([t_feat, lab, featKW])

    # getting anomalies
    anomalies = []
    for i, feat in enumerate(outliers):
        points = []
        if isinstance(feat, str):
            anom = data[feat]
            lab = feat
        else:
            anom = feat
            lab = f"Anomaly {i}"
        if outlierLabels:
            lab = outlierLabels[i]

        if useOutlierLabelArray and (len(anom) == len(x) or useOutlierLabelArray != "auto"):
            if showPoints:
                points = __edges(np.array(anom, dtype=bool).astype(int))
            anom = __index_from_vec(anom)
        else:
            if showPoints:
                points = __edges(__vec_from_index(anom, len(x[0])))
        anomalies.append([anom, lab, points])

    fig, ax = __draw_features(x, features, anomalies, ax)
    return fig, ax


def __draw_features(x: Tuple[Sequence[float], str], features: Sequence[Tuple[Sequence[float], str, Mapping[str, Any]]], outliers: Sequence[Tuple[Sequence[float], str, Sequence[float]]], ax: Union[plt.Axes, None] = None) -> Tuple[Figure, Axes]:
    """Actually plot the features with the anomalies.

    It receives the input from the higher level `plot_features` and generates the plot. 

    Args:
        x (Tuple[Sequence[float], str]): The values of x and the label for the x axis.
        features (Sequence[Tuple[Sequence[float], str, Mapping[str, Any]]]): A sequence of features to plot. Every feature is represented as the y values, a label for the legend and a dictionary of kwargs for the plotting function.
        outliers (Sequence[Tuple[Sequence[float], str, Sequence[float]]]): A sequence of anomalies to plot. Every anomaly is represented by the indexes corresponding to anomalous points, a label for the legend and the terminal points of anomalous intervals if needed.
        ax (Union[plt.Axes, None], optional): An instance of `matplotlib.axes.Axes` used to plot the graph. Defaults to None.

    Returns:
        Tuple[Figure, Axes]: Matplotlib figure and axes containing the plot.
    """
    if ax is not None:
        plt.sca(ax)
        fig = plt.gcf()
    else:
        fig = plt.figure(figsize=(8, 3))
        ax = plt.gca()

    feat_plot = []
    for y, lab, KW in features:
        feat_plot.append([plt.plot(x[0], y, **KW)[0], lab])
    out_plot = []
    if len(outliers) > 0:
        if len(outliers) == 1:
            for y, lab, KW in features:
                out_plot.append(__add_outliers(
                    x[0], y, outliers[0], ["r", "maroon"]))
        elif len(features) == 1:
            for outl in outliers:
                out_plot.append(__add_outliers(x[0], features[0][0], outl))
        else:
            for curve, outl in zip(features, outliers):
                out_plot.append(__add_outliers(
                    x[0], curve[0], outl, ["r", "maroon"]))

    handles, labels = __determine_handles_labels(
        len(features), len(outliers), feat_plot, out_plot)

    plt.xlabel(x[1])
    if len(features) == 1:
        plt.ylabel(features[0][1])
    else:
        plt.ylabel("Feature")
    plt.legend(handles, labels, loc=0)
    return fig, ax


def __add_outliers(x: Sequence[float], y: Sequence[float], outl: Tuple[Sequence[float], str, Sequence[float]], colors: Tuple[Any, Any] = None) -> Tuple[Line2D, str, PathCollection]:
    """Overlays the anomalous points over the features.

    Args:
        x (Sequence[float]): x coordinates of the points.
        y (Sequence[float]): y coordinates of the points.
        outl (Tuple[Sequence[float], str, Sequence[float]]): The anomalies to plot represented by the indexes corresponding to anomalous points, a label for the legend and the terminal points of anomalous intervals if needed.
        colors (Tuple[Any, Any], optional): Colours of outliers line and markers, useful to force a single colour for all the outliers. Defaults to None.

    Returns:
        Tuple[Line2D, str, PathCollection]: A tuple containing the information for adding the anomalies to the legend (handle for the anomaly line, anomaly label, handle for the points markers).
    """
    o, lab, pts = outl
    if colors is None:
        c1, c2 = None, None
    else:
        c1, c2 = colors
    new_y = np.full_like(y, np.nan)
    new_y[o] = y[o]
    t_out_plot, = plt.plot(x, new_y, color=c1)
    if len(pts):
        if c2 is None:
            c2 = [a*0.7 for a in to_rgb(t_out_plot.get_color())]
        t_point_plot = plt.scatter(
            x[pts], y[pts], color=c2, marker="d", s=20)
    else:
        t_point_plot = None
    return [t_out_plot, lab, t_point_plot]


def __determine_handles_labels(len_feat: int, len_outl: int, feat_plot: Sequence[Tuple[Line2D, str]], out_plot: Sequence[Tuple[Line2D, str, PathCollection]]) -> Tuple[Sequence, Sequence[str]]:
    """Determine the handles and labels for the legend associated to `__draw_features`.

    The number of elements and their content is determined based on the kind of plot as described for `plot_features`. To reduce the space taken up by the legend, uses a single entry for feature and anomaly when these are paired.

    Args:
        len_feat (int): Number of features plotted.
        len_outl (int): Number of anomalies plotted.
        feat_plot (Sequence[Tuple[Line2D, str]]): A sequence of tuples containing a `Line2D` and a label for every feature plotted.
        out_plot (Sequence[Tuple[Line2D, str, PathCollection]]): A tuple containing the information for adding the anomalies to the legend (handle for the anomaly line, anomaly label, handle for the points markers).

    Returns:
        Tuple[Sequence, Sequence[str]]: A tuple containing the sequences of handles and labels.
    """
    handles = [[pl[0]] for pl in feat_plot]
    labels = [pl[1] for pl in feat_plot]

    if out_plot:
        if len_outl == len_feat and len_feat != 1:
            for i, op in enumerate(out_plot):
                ha, la, pt = op
                ph = Line2D([0, 0], [0, 0], dashes=[8, 8], color="red")
                if pt is not None:
                    handles[i].extend([ph, pt])
                else:
                    handles[i].append(ph)
                labels[i] = "\n".join([labels[i], la])
        elif len_outl > 1:
            for ha, la, pt in out_plot:
                handles.append([ha])
                if pt is not None:
                    handles[-1].append(pt)
                labels.append(la)
        else:
            ha, la, pt = out_plot[0]
            handles.append([ha])
            if pt is not None:
                handles[-1].append(pt)
            labels.append(la)
    handles = [ha[0] if len(ha) == 1 else tuple(ha) for ha in handles]

    return handles, labels


def __edges(labels: Sequence[bool]) -> Sequence[int]:
    """Identify the edges of intervals of consecutive `True` values.

    The returned array contains the indices of the first and last `True` value of every interval and the indices of isolated ones.

    Args:
        labels (Sequence[bool]): An array containing `True` and `False` labels.

    Returns:
        Sequence[bool]: The indices of the edges.
    """
    blur = np.diff(labels)
    sharp = labels.copy()
    sharp[1:-1] = blur[:-1]-blur[1:]
    return np.where(sharp > 0)[0]


def __check_columns(data: pd.DataFrame, y: Sequence[str]) -> None:
    """Verify that a list of names are among the columns of a `pd.DataFrame`.

    Useful to give a meaningful error in case several are missing.

    Args:
        data (pd.DataFrame): The `DataFrame` to check.
        y (Sequence[str]): The desired columns.
    """
    missy = []
    for feature in y:
        if feature not in data:
            missy.append(feature)
    misstr = ", ".join(str(feature) for feature in missy)
    assert len(missy) == 0, "None of: "+misstr+" in data."


def __to_sequence(y: Any) -> Sequence[Any]:
    """Ensure that an object is a sequence of sequences.

    It receives a sequence or none as input and ensures that the output is an empty list or a sequence of sequences.

    If the input is not a sequence is returned as is for future error management.

    Args:
        y (Any): Object that must become a sequence of sequences.

    Returns:
        Sequence[Any]: A sequence of sequences.
    """
    if y is None or (hasattr(y, "__len__") and len(y) == 0):
        return []
    if isinstance(y, str) or isinstance(y, dict) or (hasattr(y, "__len__") and not hasattr(y[0], "__len__")):
        return [y]
    return y


def __index_from_vec(vec: Sequence[bool]) -> Sequence[int]:
    """Return the indexes of the true values.

    Args:
        vec (Sequence[bool]): Binary array containing class labels.

    Returns:
        Sequence[int]: Array containing the indices of the `True` elements.
    """
    return np.where(vec)[0]


def __vec_from_index(ind: Sequence[int], size: int) -> Sequence[bool]:
    """Return an array of labels.

    Args:
        ind (Sequence[int]): The indices of the `True` elements.
        size (int): The length of the output array.

    Returns:
        Sequence[bool]: An array containing ones in the positions pointed by `ind`.
    """
    vec = np.zeros(size)
    vec[ind] = 1
    return vec


def show_trajectories2D(x: Union[Sequence, str], y: Union[Sequence, str], t: Union[Sequence, str, None] = None, outliers: Union[Sequence[Sequence[int]], None] = None, useOutlierLabels: bool = False, data: Union[pd.DataFrame, None] = None, tistime: bool = False, outlier_names: Union[str, Sequence[str], None] = None, cbarName: Union[str, None] = None, ax: Union[plt.Axes, None] = None, vmax: Union[float, None] = None, vmin: Union[float, None] = None, forceCbar: bool = False, equalAspectRatio: bool = False) -> Tuple[Figure, Axes]:
    """Plot trajectories in 2D with color coding and anomalies.

    Plot trajectories in 2D encoding a third dimension (e.g, time, speed) as color. It is possible to plot a number of detected anomalous points over the trajectory. Up to four different anomalies are supported.

    The function provides special support for timestamp based color coding.

    It is possible to plot more than one trace on the same plot by passing an instance of `matplotlib.axis.Axis` to the function. In this case a new colorbar is not generated (unless explicitly requested). To ensure that all the traces share the same color coding, pass the vmax and vmin limits of the colorbar.

    Args:
        x (Union[Sequence, str]): x coordinates of the trajectory or name of a column in `data`.
        y (Union[Sequence, str]): y coordinates of the trajectory or name of a column in `data`.
        t (Union[Sequence, str, None], optional): Values for the colour coding or name of a column in `data`. Defaults to None.
        outliers (Union[Sequence[Sequence[int]], None], optional): A sequence of sequences of indexes representing the anomalous points. Defaults to None.
        useOutlierLabels (bool, optional): If this flag is `True` the outliers sequences are considered to be labels for every point marking th anomalies with a true value. Defaults to False.
        data (Union[pd.DataFrame, None], optional): If x, y or t are strings, the function looks for data in the columns of this `DataFrame`. Defaults to None.
        tistime (bool, optional): A flag to signal that the t sequence contains times for proper formatting. Defaults to False.
        outlier_names (Union[str, Sequence[str], None], optional): Sequence of strings providing names for the anomalies for the legend. Use the special value "data" to signal that the `outliers` contains the names of columns in `data`. Defaults to None.
        cbarName (Union[str, None], optional): The label for the colorbar describing the color coding. Defaults to None.
        ax (Union[plt.Axes, None], optional): An instance of `matplotlib.axis.Axis` to plt into. Defaults to None.
        vmax (Union[float, None], optional): Upper limit of the colorbar. Defaults to None.
        vmin (Union[float, None], optional): The lower limit of the colorbar. Defaults to None.
        forceCbar (bool, optional): Force the creation of a colorbar even when the `ax` argument is passed. By default when the `ax` is passed a colorbar is not generated to avoid having multiple colorbars in the same plot. Defaults to False.
        equalAspectRatio (bool, optional):   Defaults to False.

    Returns:
        Tuple[Figure, Axes]: Matplotlib figure and axes containing the plot.
    """
    outliers = __to_sequence(outliers)
    assert len(outliers) < 4, "Up to four kind of anomalies can be displayed."
    if len(outliers)==0:
        assert outlier_names is None, "Can't pass outlier names without specifying outlier_indexes or outlier_labels."
    _outliers = {}
    if data is not None:
        assert x in data, str(x)+" == x not a column in data."
        assert y in data, str(x)+" == y not a column in data."
        if isinstance(t, str) and t in data:
            t = data[t]
        if outlier_names == "data":
            __check_columns(data, outliers)
            for feature in outliers:
                _outliers[feature] = __index_from_vec(data[feature])
        xName = x
        yName = y
        x = data[x]
        y = data[y]
    else:
        assert outlier_names != "data", "Can't ask for outlier names in data without passing data."
        assert len(x) == len(y), "x and y must have the same length."
        xName = "x"
        yName = "y"
    outlier_names = __to_sequence(outlier_names)
    if len(outlier_names) > 0:
        assert len(outlier_names) == len(outliers), "If you provide names for anomalies exactly one per anomaly must be passed. len(outlier_names)=="+str(len(outlier_names))
        for name, outl in zip(outlier_names, outliers):
            _outliers[name] = outl
    else:
        if len(outliers) > 0:
            for i, outl in enumerate(outliers):
                _outliers[f"Anom {i}"] = outl

    if isinstance(t, str) and t == "range":
        t = np.arange(len(x))
        tistime = False

    if useOutlierLabels:
        _outliers = {k: __index_from_vec(v) for k, v in _outliers}

    return __2Dplot(x, y, t, _outliers, tistime, xName, yName, cbarName, ax, vmax, vmin, forceCbar, equalAspectRatio)


def __2Dplot(x: Sequence[float], y: Sequence[float], t: Sequence[float], outliers: Mapping[str, Sequence[float]], tistime: bool, xName: Union[str, None] = None, yName: Union[str, None] = None, cbarName: Union[str, None] = None, ax: Union[plt.Axes, None] = None, vmax: Union[float, None] = None, vmin: Union[float, None] = None, forceCbar: bool = False, equalAspectRatio: bool = False) -> Tuple[Figure, Axes]:
    """Does the actual plotting for `show_trajectories2D`.

    Args:
        x (Sequence[float]): x values.
        y (Sequence[float]): y values.
        t (Sequence[float]): Values for color coding.
        outliers (Mapping[str, Sequence[float]]): Dictionary containing anomalies labels and indexes.
        tistime (bool): Determine id the color coding id to be treated as a timestamp.
        xName (Union[str, None], optional): Name for the x axis. Defaults to None.
        yName (Union[str, None], optional): Name for the y axis. Defaults to None.
        cbarName (Union[str, None], optional): Name for the colorbar. Defaults to None.
        ax (Union[plt.Axes, None], optional): An instance of `matplotlib.axis.Axis` to plt into. Defaults to None.
        vmax (Union[float, None], optional): Upper limit of the colorbar. Defaults to None.
        vmin (Union[float, None], optional): The lower limit of the colorbar. Defaults to None.
        forceCbar (bool, optional): Force the creation of a colorbar even when the `ax` argument is passed. Defaults to False.
        equalAspectRatio (bool, optional): Force equal aspect ratio of the plot, prevents stretching of spatial data.. Defaults to False.

    Returns:
        Tuple[Figure, Axes]: Matplotlib figure and axes containing the plot.
    """
    if ax is not None:
        plt.sca(ax)
        fig = plt.gcf()
    else:
        fig = plt.figure()

    if len(outliers) > 0:
        for outl, style in zip(outliers.items(), styles):
            plt.scatter(x[outl[1]], y[outl[1]], s=np.full_like(
                outl[1], style[0]), color=style[1],  marker=style[2], label=outl[0])
        plt.legend()

    if tistime and isinstance(vmax, pd.Timestamp):
        vmax = vmax.timestamp()
        fact = 18 - int(np.log10(vmax))
        vmax *= np.power(10, fact)
    if tistime and isinstance(vmin, pd.Timestamp):
        vmin = vmin.timestamp()
        fact = 18 - int(np.log10(vmin))
        vmin *= np.power(10, fact)
    plt.scatter(x, y, s=np.full_like(x, 7), c=t,
                cmap='jet', zorder=0, vmax=vmax, vmin=vmin)
    if equalAspectRatio:
        ax.axis("equal")
    plt.xlabel(xName)
    plt.ylabel(yName)
    if hasattr(t, "__len__") and len(t) == len(x):
        if ax is None or forceCbar:
            ax = plt.gca()
            cbar = plt.colorbar()

            if isinstance(cbarName, str):
                cbar.set_label(cbarName)
            if tistime:
                if cbarName is None:
                    cbar.set_label('Time')
                tk = cbar.get_ticks()
                cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(tk))
                cbar.ax.set_yticklabels(
                    list(map(str, pd.to_datetime(tk).time)))
        else:
            if (vmin is None) or (vmax is None):
                warnings.warn(
                    "When providing the axes directly a new colorbrbar is not generated. It is adviced to provide vmax and vmin limits for the colors to avoid inconsistent color mapping.")
    else:
        ax = plt.gca()

    return fig, ax
