"""This submodule provides a set of utility function for anomaly dectection on 2D particle traces.

The main content of this submodule is the `preprocess` function. The minimum data required is just the list of x and y positions of the particle. However, it offers much more flexibility allowing for interpolation of missing data with improved smoothing.

The other functions of this submodule are focused on reporting to help the user getting an idea of the content of their dataset.

Preprocessing a trace can be as easy as:

```python
import pandas as pd
from candela.particles2d import preprocess
df = pd.read_csv("my_trace.csv")
trace = preprocess(df, track_id=0)
```
In more common scenarios where the dataframe contain multiple traces, when they are indexed by a `track_id` column, the full dataset is preprocessed with:

```python
df = pd.read_csv("my_dataset.csv")
preprocessed_traces = df.groupby("track_id", as_index=False).apply(preprocess, fps=24, scale=2)
```

Where we also specify a framerate of the acquisition of 24 frames per second and a factor 2 in the spatial scale (1 unit in the data corresponds to 2 m).

Even before preprocessing the whole dataset, it may be convenient to get a first overview of the data. The function `anomaly_summary` serves this purpose:

```python
from candela.particles2d import anomaly_summary
df = pd.read_csv("my_dataset.csv")
sample = df[df.track_id==sample_id]
anomaly_summary(sample, sample_id, fps=24, scale=2)
```

"""
import pandas as pd
import numpy as np
import warnings
from typing import Sequence, Tuple
from .functions import isolation_forest, simple_threshold
from .plotting import plot_features, show_trajectories2D
import matplotlib.pyplot as plt

__pdoc__ = {}
__pdoc__["__interpolate_xy"] = True
__pdoc__["__smooth_speeds"] = True
__pdoc__["__angular_features"] = True


def preprocess(df: pd.DataFrame, track_id: int = None, sequential_frames: bool = False, filterVel: bool = False, window: int = 3, fps: float = 10, scale: float = 1, kmh: bool = True, startTime: float = 1356088042.042042) -> pd.DataFrame:
    """Preprocess the tracks of particles in a 2D space for further analysis and anomaly detection.

    The input DataFrame must contain at least the following columns:

        - x, y: coordinates of the particle for every time point.

    It is suggested to include also:
        
        - frame_id: time point id of the measured values, allowing for missing frames;
        - track_id: particle id of the measured values.

    The preprocessind includes three steps:

    1. scaling of the positions and interpolation in case of missing time points (assuming constant speed);
    2. computation of the instantaneous speed, smoothing with double pass exponential window;
    3. reconstruction of positions given the smoothed speed and computation of additional statistics as the heading angle and acceleration.

    Args:
        df (pd.DataFrame): Input data
        track_id (int, optional): If provided, supersedes the eventual track_id column in df.
        sequential_frames (bool, optional): If True ignores frame_id and reindexes the frames. Defaults to False.
        filterVel (bool, optional): If True replaces the computed headings with NaN when the speed is too low. Defaults to False.
        window (int, optional): Decay length in time frames of the smoothing applied to speeds. Defaults to 3.
        fps (float, optional): Frames per second of data, needed to get actual speeds. Defaults to 10.
        scale (float, optional): Scale of the positions, assumed equal over x and y. Defaults to 1.
        kmh (bool, optional): Use km/h as unit for speeds. Defaults to True.
        startTime (float, optional): Timestamp of the first frame. Defaults roughly to the end of the world according to the Mayas.

    Returns:
        pd.DataFrame: Interpolated and smoothed data with additional measures (heading, speed, total speed, angular velocity).
    """
    assert len(df) > 0
    assert track_id is not None or "track_id" in df, "You must provide a track_id"
    assert sequential_frames or "frame_id" in df
    assert "x" in df
    assert "y" in df
    assert window > 0

    if "track_id" in df:
        assert len(df.track_id.unique(
        )) == 1, "The dataframe seems to contain more than one track (multiple unique values is 'track_id' column)."

    if track_id is None:
        track_id = df.track_id.unique()[0]

    if sequential_frames:
        fid_min = 0
        fid_max = len(df)
        steps = fid_max

        # compute interpolated temporary positions
        tx, ty = df.x*scale, df.y*scale

    else:
        fid_min = df.frame_id.min()
        fid_max = df.frame_id.max()
        steps = fid_max - fid_min + 1

        # compute interpolated temporary positions
        tx, ty = __interpolate_xy(df, scale)

    # compute smoothed velocities and residuals
    vx, res_x = __smooth_speeds(tx, window)
    vy, res_y = __smooth_speeds(ty, window)

    # compute smoothed positions
    x = np.full(len(tx), tx[0]) + np.cumsum(vx)
    y = np.full(len(ty), ty[0]) + np.cumsum(vy)

    # get true velocities
    vx *= (3.6 if kmh else 1)*fps
    vy *= (3.6 if kmh else 1)*fps
    vtot = np.sqrt(vx*vx+vy*vy)

    # get true accelerations in m/s^2 or km/h/s
    ax = np.zeros_like(vx)
    ax[1:] = np.diff(vx)*fps
    ay = np.zeros_like(vy)
    ay[1:] = np.diff(vy)*fps
    atot = np.sqrt(ax*ax+ay*ay)

    # compute angular features
    angle, angular_velocity = __angular_features(
        vx, vy, vtot, filterVel, 0.1 if kmh else 0.03)

    # compute total residual
    res = np.zeros_like(tx)
    res[1:] = np.sqrt(np.power(res_x, 2)+np.power(res_y, 2)) * \
        (3.6 if kmh else 1)*fps

    # compute true time for every time frame
    frame_id = fid_min+np.arange(steps)
    true_time = pd.to_datetime(
        startTime+frame_id/fps, unit='s')

    ret_df = pd.DataFrame({"track_id": np.full_like(x, track_id), "frame_id": frame_id, "x": x, "y": y,
                          "vx": vx, "vy": vy, "vtot": vtot, "ax": ax, "ay": ay, "atot": atot, "angle": angle, "angular_velocity": angular_velocity, "residual": res, 'true_time': true_time})

    return ret_df


def __interpolate_xy(df: pd.DataFrame, scale: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Given a list of observations for an ID, interpolates linearly in the missing frames.

    Args:
        df (pd.DataFrame): Dataframe containing the observations
        scale (float, optional): Scale for the distances. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays containing the interpolated positions
    """
    fid_min = df.frame_id.min()
    fid_max = df.frame_id.max()
    steps = fid_max - fid_min + 1

    tx = np.zeros(steps)
    ty = np.zeros(steps)
    last = fid_min - 1
    for idx, row in df.iterrows():
        fid = int(row.frame_id)
        ind = fid - fid_min
        tx[ind] = row.x
        ty[ind] = row.y
        if fid > last + 1:
            lind = last-fid_min
            dt = ind-lind
            dx = (tx[ind]-tx[lind])/dt
            dy = (ty[ind]-ty[lind])/dt
            for i in range(1, dt):
                tx[lind+i] = tx[lind]+dx*i
                ty[lind+i] = ty[lind]+dy*i
        last = fid
    tx *= scale
    ty *= scale

    return tx, ty


def __smooth_speeds(pos: Sequence, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothing with exponentially weighted mean.

    To reduce the bias the function is applied once forwards and once backwards. This may introduce dependency from the future in causal estimation.

    Args:
        pos (Sequence): Sequence to be smoothed
        window (int): Amplitude of the smoothing window (decay of the weight in terms of frames).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A pair of arrays containing the smoothed values and the residuals.
    """
    tv = np.diff(pos)
    v = np.zeros_like(pos)
    v[1:] = pd.Series(tv).ewm(window, min_periods=1).mean(
    ).iloc[::-1].ewm(window, min_periods=1).mean().iloc[::-1]

    return v, v[1:]-tv


def __angular_features(vx: np.ndarray, vy: np.ndarray, vtot: np.ndarray = None, filterVel: bool = False, threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Computes angular features related to the heading.

    Angles are in degrees in (-180, +180). Angular speed is computed such that moving from -179 to +179 gives a difference of 2°.

    Args:
        vx (np.ndarray): Velocity on x.
        vy (np.ndarray): Velocity on y
        vtot (np.ndarray, optional): Total velocity, if not provided and filterVel is True must be recomputed. Passing this argument may speed up the computation. Defaults to None.
        filterVel (bool, optional): If True replaces the computed headings with NaN when the speed is too low. Defaults to True.
        threshold (float, optional): Threshold for the total speed below which the angle is filtered out. Defaults to 0.1 assuming km/h.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays containing the heading and the angular speed (degrees per frame).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        angle = np.arctan(vy/vx)*180/np.pi
    angle[vx < 0] += 180*np.sign(vy[vx < 0])
    angle[vx == 0] = 90*np.sign(vy[vx == 0])
    angular_velocity = np.zeros_like(vx)
    angular_velocity[1:] = np.diff(angle)
    angular_velocity[angular_velocity > 180] -= 360
    angular_velocity[angular_velocity < -180] += 360
    angular_velocity[np.isnan(angular_velocity)] = 0
    if filterVel:
        if vtot is None:
            vtot = np.sqrt(vx*vx+vy*vy)
        angle[vtot < threshold] = np.nan
        angular_velocity[vtot < threshold] = np.nan

    return angle, angular_velocity


def __print_viewer_info(df: pd.DataFrame):
    """This function provides some basic information extracted from the output of preprocess.

    This might be useful in case a track viewer is used.

    Args:
        df (pd.DataFrame): Outupt of the preprocess function.
    """
    try:
        import tracking_viewer.consts as constpack
        path = constpack.__file__
    except ImportError:
        warnings.warn("No viewer installed, this is general info")
        path = "**missing package**"
    print(f"Paste this at the end of {path} and restart viewer:\n#############\nINTERESTING_ID = ",
          df.track_id.min(), "\nINTERESTING_START_SEC = ", df.true_time.min().timestamp(), "\n#############", sep="")


def basic_stats(df: pd.DataFrame):
    """This function prints information extracted from the output of preprocess.

    Args:
        df (pd.DataFrame): Output of the preprocess function.
    """
    print(f"Track_id: {df.track_id.min()}.")
    print(
        f"Total track length: {df.true_time.max().timestamp()-df.true_time.min().timestamp():.3} s")
    print(f"Min x: {df.x.min():.4}, Max x: {df.x.max():.4}")
    print(f"Min y: {df.y.min():.4}, Max y: {df.y.max():.4}")
    print(
        f"Sepeed: Max {df.vtot.max():.4}, Min {df.vtot.min():.4}, Mean {df.vtot.mean():.4}+/-{df.vtot.std():.4}")


def anomaly_summary(df: pd.DataFrame, track_id: int = None, **kwargs):
    """Utility function providing an overview of the anomalies in a track.

    Args:
        df (pd.DataFrame): pd.DataFrame containing the data of a single track.
        track_id (int, optional): Id of the processed track if df does not contain a track_id column. Defaults to None.
        kwargs: Arguments passed to the `preprocess` function, see `preprocess`.
    """
    prep = preprocess(df, track_id, **kwargs)
    basic_stats(prep)
    fig, ax = plot_features("frame_id", ["vx", "vtot", "vy"], featureLabels=[r'$V_{x}$', r'$V_{y}$', r'$V_{total}$'], featureKW=[
                            {"linewidth": .75, "linestyle": 'dashed'}, {"linewidth": .75, "linestyle": 'dashed'}, {}], data=prep)
    plt.show()
    fig, ax = plot_features("frame_id", ["angle", "angular_velocity"], featureLabels=[
                            r'$\alpha$', r'$\omega$'], featureKW=[{"linewidth": .75, "linestyle": 'dashed'}, {}], data=prep)
    plt.show()

    angular_velocity_isolation_forest = isolation_forest(
        prep.angular_velocity, absolute=True)
    angular_velocity_threshold = simple_threshold(
        prep.angular_velocity, threshold=25, absolute=True)
    fig, ax = plot_features('true_time', 'angular_velocity',
                            outliers=angular_velocity_isolation_forest, data=prep)
    ax.set_title(r"$\omega -$ Isolation Forest")
    plt.show()
    perc_outl_if = 100*len(angular_velocity_isolation_forest)/len(prep)
    print(
        f"Isolation forest: {perc_outl_if:.4}% ({len(angular_velocity_isolation_forest)} points.)")
    fig, ax = plot_features('true_time', 'angular_velocity',
                            outliers=angular_velocity_threshold, data=prep, showPoints=True)
    ax.set_title(r"$\omega -$ Simple Threshold")
    plt.show()
    perc_outl_threshold = 100*len(angular_velocity_threshold)/len(prep)
    print(
        f"Isolation forest: {perc_outl_threshold:.4}% ({len(angular_velocity_threshold)} points.)")

    fig = show_trajectories2D('x', 'y', 'true_time', outliers=[angular_velocity_isolation_forest, angular_velocity_threshold], data=prep, tistime=True, outlier_names=[
                              "angular_velocity_isolation_forest", "angular_velocity_threshold"])
    plt.title("Angular anomalies")
    plt.show()
