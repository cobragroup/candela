"""This submodule provides a set of function for anomaly dectection.

The content of this submodule is divided to two parts: a group of functions for anomaly detection, a class for managing anomaly detection based on a several features at once through an higher level processing.

The functions are of varying complexity moving from the `simple_threshold` working only on unidimensional data to higher complexity tools as the `one_class_svm`. In particular `simple_threshold`, `double_threshold` and `tukeys_method` work only on 1D series while `isolation_forest`, `one_class_svm`, `robust_covariance` and `local_outlier_factor` work also on nD data.

The scatter_cluster class is specialised in the use with data from particles moving in 2D as produced by particles2d.preprocess. It is however easy to apply to other kind of data providing a list of relevant features. The class offers methods for segmenting the data extracting useful statistics with a sliding window.

Given generic time series or a trace (possibly preprocessed, see `particles2d.preprocess`) identifying anomalies can be as easy as:

```python
import pandas as pd
from candela.functions import local_outlier_factor
from candela.particles2d import preprocess

df = pd.read_csv("my_trace.csv")
trace = preprocess(df, track_id=0)

outliers_velocity_LOF = local_outlier_factor(trace.vtot)
```

When using a `scatter_cluster` more steps are involved:

```python
from candela.functions import scatter_cluster

segmented = scatter_cluster.segment(trace)
sc_model = scatter_cluster("IsolationForest")
sc_model.fit(segmented, feature_list="all")
sc_model.predict(segmented)
```

The currently available kernels for the `scatter_cluster` are: IsolationForest, OneClassSVM, RobustCovariance. It is possible to add new kernels (provided they expose the `fit` and `predict` methods) by updating the `scatter_cluster.kernels` attribute.

For more advanced segmentation options, see the documentation for `scatter_cluster.segment_trace`.

"""
from typing import Sequence, Union, Any, Tuple, Callable
import numpy as np
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
__pdoc__ = {}
__pdoc__["__add_tails"] = True
__pdoc__["__simple_threshold"] = True
__pdoc__["scatter_cluster._kernels"] = True
__pdoc__["scatter_cluster._base_features"] = True


def simple_threshold(data: Sequence, absolute: bool = False, smaller: bool = False, exclude_nan: bool = True, fraction: float = 0.01, number: int = None, threshold: float = None) -> Sequence:
    """This function identifies anomalies by simple thresholding.

    There are three different ways to provide the threshold, if more than one method is provided, the first one in this succession is applied: value, number of anomalies, fraction of anomalies. It is possible to work with absolute values and to exclude NaNs, this mainly affect the method passing fraction of anomalies by possibly reducing the total number of points.

    It works only on unidimensional data.

    Args:
        data (Sequence: list, array, pd.Series): The data containing anomalies. Must be a 1D sequence.
        absolute (bool, optional): Activates the use of absolute values. Defaults to False.
        smaller (bool, optional): Values below the threshold (or the lowest fraction) are considered anomalies. Defaults to False.
        exclude_nan (bool, optional): If True, NaN values are excluded when considering the fraction of data points to report. Defaults to True.
        fraction (float, optional): Fraction of data points considered as anomaly, lowest priority. Defaults to 0.01.
        number (int, optional): Number of data points considered as anomaly, intermediate priority. Defaults to None.
        threshold (float, optional): Threshold above (below) which data is considered anomaly, highest priority. Defaults to None.

    Returns:
        Sequence: An array containing the indexes of the anomalies.
    """
    return __simple_threshold(data, absolute, smaller, exclude_nan, fraction, number, threshold)[0]


def __simple_threshold(data: Sequence, absolute: bool = False, smaller: bool = False, exclude_nan: bool = True, fraction: float = 0.01, number: int = None, threshold: float = None) -> Tuple[Sequence, float, Sequence]:
    """
    See the documentation for `simple_threshold`.

    Returns:
        Tuple[Sequence, float, Sequence]: An array containing the indexes of the anomalies, but also the effective threshold (not known beforehand when using fraction or number of points) and the data used for the actual thresholding (possibly the absolute value or the negation of the input data).
    """
    assert len(data) > 0, "Empty sequence passed."
    _data = np.squeeze(
        np.abs(data)) if absolute else np.squeeze(np.array(data))
    assert len(_data.shape) == 1, "This function only works with 1-D data."
    if exclude_nan:
        full = _data.copy()
        _data = full[np.isfinite(full)]

    if smaller:
        _data = -_data
    if threshold is None:
        if number is not None:
            assert number > 0, "The number of requested anomalies must be greater than 0."
            if number > len(_data):
                warnings.warn("Asking for more anomalies than data.")
            fraction = min(number/len(_data), 1)

        assert 0 < fraction <= 1, "The fraction of points identified as anomalies must be in (0, 1]."
        threshold = np.quantile(_data, fraction)
    elif smaller:
        threshold = -threshold

    anomalies = _data > threshold
    if exclude_nan:
        full_anomalies = np.zeros_like(full)
        full_anomalies[np.isfinite(full)] = anomalies
        anomalies = full_anomalies

    return np.where(anomalies == 1)[0], threshold, _data


def double_threshold(data: Sequence[float], secondThreshold: float, secondSmaller: bool = False, useNumpoints: bool = False, returnNoise: bool = False, maxSpacing: int = 10, **kwargs) -> Union[Sequence[int], Tuple[Sequence[int], Sequence[int]]]:
    """This function uses a double threshold and a simple heuristics to identify anomalies when expected to appear in groups.

    It first applies a `simple_threshold` (pass the arguments as for the simple_threshold function as key word arguments). 
    As a second step it joins all the points above threshold spaced no more than maxSpacing points marking them all as potential anomalies. 
    Finally it identifies as anomalies only the groups with at least secondThreshold points (if useNumpoints) or where the sum of the (absolute) values of the function iver the points of the group is greater (smaller) than secondThreshold.

    It works only on unidimensional data.

    Args:
        data (Sequence[float]: list, array, pd.Series): The sequence of data points. Must be a 1D sequence.
        secondThreshold (float): The sum of the absolute values of the sequence point in a group to qualify as anomaly.
        secondSmaller (bool, optional): If the sum shall be below threshold to qualify as anomaly. Defaults to False.
        useNumpoints (bool, optional): If to use just the number of points. Defaults to False.
        returnNoise (bool, optional): If to return a second array with the indexes of the points candidate as anomaly but not qualifying. Defaults to False.
        maxSpacing (int, optional): The maximum distance between points above the first threshold to be considered part of the same group. Half of this interval is always added at the extremes of the group. Defaults to 10.
        kwargs: Arguments for `simple_threshold`.

    Returns:
        Union[Sequence[int],Tuple[Sequence[int], Sequence[int]]]: An array containing the indexes of the anomalies. If returnNoise == True also the points candidate as anomaly but not qualifying are returned as a second array.
    """
    above_spThr, firstThreshold, data = __simple_threshold(data, **kwargs)

    # joining close by anomalies
    filler = []
    last = above_spThr[0]
    for i in above_spThr:
        if last+1 < i <= last+maxSpacing:
            filler.append(np.arange(last+1, i))
        last = i
    if len(filler):
        above_spThr = np.sort(np.concatenate([above_spThr, *filler]))

    pieces = []
    pieces_sizes = []
    tmp_piece = []
    for i in above_spThr:
        if tmp_piece and tmp_piece[-1] < i-1:
            if not useNumpoints:
                tmp_piece, tmp_size = __add_tails(
                    data, firstThreshold/2, int(maxSpacing/2), tmp_piece)
            pieces.append(tmp_piece)
            pieces_sizes.append(tmp_size)
            tmp_piece = []

        tmp_piece.append(i)
    if not useNumpoints:
        tmp_piece, tmp_size = __add_tails(
            data, firstThreshold/2, int(maxSpacing/2), tmp_piece)
    pieces.append(tmp_piece)
    pieces_sizes.append(tmp_size)

    if secondSmaller:
        pieces_sizes = [-ps for ps in pieces_sizes]
        secondThreshold = -secondThreshold

    if useNumpoints:
        pieces_sizes = [len(piece) for piece in pieces]

    tmp_good = []
    tmp_bad = []
    for i, size in enumerate(pieces_sizes):
        if size > secondThreshold:
            tmp_good.append(pieces[i])
        else:
            tmp_bad.append(pieces[i])
    goodIndex = np.concatenate(tmp_good) if len(
        tmp_good) else np.array([], dtype=int)
    if returnNoise:
        return goodIndex, np.concatenate(tmp_bad) if len(tmp_bad) else np.array([], dtype=int)

    return goodIndex


def __add_tails(data: Sequence[float], threshold: float, tailSize: int, piece: Sequence[int]) -> Tuple[Sequence[int], float]:
    """Adds tails to the groups identified in the process of double thresholding.

    Adds up to tailSize points, given that they are not smaller than a threshold. Usually the tail is long half maxSpacing and the threshold is half the first threshold.

    Args:
        data (Sequence[float]): Sequence under processing
        threshold (float): This should be more permissive that the original threshold.
        tailSize (int): Maximum number of points to add on both ends of `piece`.
        piece (Sequence[int]): Group of points identified as outliers.

    Returns:
        Tuple[Sequence[int], float]: Indexes of the candidate anomalies in the group, total contributions of all points in the group of candidate anomalies.
    """
    head = []
    for i in range(piece[0], piece[0]-tailSize, -1):
        head.append(i)
        if data[i] < threshold or i == 0:
            break

    tail = []
    for i in range(piece[0], piece[0]+tailSize):
        tail.append(i)
        if data[i] < threshold or i == len(data)-1:
            break

    t_piece = head[::-1]+piece+tail
    pieceSize = np.sum(np.abs(data[t_piece]))
    return t_piece, pieceSize


def tukeys_method(data: Sequence, absolute: bool = False, anomalyScale: float = 1.5) -> Sequence:
    """This function identifies anomalies using the Tukey's algorithm.

    This is a reimplementation of the _Tukey's method_ (Tukey, John W (1977). "Exploratory Data Analysis". Addison-Wesley Publishing Company). It works only on unidimensional data.

    Args:
        data (Sequence: list, array, pd.Series): The data containing anomalies. Must be a 1-D sequence.
        absolute (bool, optional): Activates the use of absolute values. Defaults to False.
        anomalyScale (float, optional): The factor multiplying the interquartile distance to determine the anomalies. Defaults to 1.5.

    Returns:
        Sequence: An array containing the indexes of the anomalies.
    """
    assert len(data) > 0, "Empty sequence passed."
    _data = np.squeeze(
        np.abs(data)) if absolute else np.squeeze(np.array(data))
    assert len(_data.shape) == 1, "This function only works with 1-D data."

    q1 = np.quantile(_data, 0.25)
    q3 = np.quantile(_data, 0.75)
    iqr = q3-q1
    inner_fence = anomalyScale*iqr

    # inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence

    return np.where((np.array(_data <= inner_fence_le) | np.array(_data >= inner_fence_ue)))[0]


def isolation_forest(data: Sequence, absolute: bool = False) -> Sequence:
    """This function identifies anomalies using the Isolation Forest algorithm whith a suitable set of parameters.

    Uses the _Isolation Forest_ (Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation forest.” Data Mining, 2008. ICDM’08. Eighth IEEE International Conference). It works also on multidimensional data.

    Args:
        data (Sequence: list, array, pd.Series): The data containing anomalies.
        absolute (bool, optional): Activates the use of absolute values. Defaults to False.

    Returns:
        Sequence: An array containing the indexes of the anomalies.
    """
    assert len(data) > 0, "Empty sequence passed."
    _data = np.squeeze(np.array(data))
    if absolute:
        _data = np.abs(_data)

    timeSeries = _data[np.isfinite(_data)]
    if timeSeries.ndim == 1:
        timeSeries = timeSeries.reshape(-1, 1)

    clf = IsolationForest(random_state=42)
    clf.fit(timeSeries)
    outlier_labels = np.where(clf.predict(timeSeries) == -1, 1, 0)
    anomalies = np.zeros_like(_data)
    anomalies[np.isfinite(_data)] = outlier_labels

    return np.where(anomalies == 1)[0]


def one_class_svm(data: Sequence, absolute: bool = False) -> Sequence:
    """This function identifies anomalies using the One-Class SVM algorithm with a suitable set of parameters.

    Uses _One-class Support Vector Machine_ (Estimating the support of a high-dimensional distribution Schölkopf, Bernhard, et al. Neural computation 13.7 (2001): 1443-1471.). It works also on multidimensional data.

    Args:
        data (Sequence: list, array, pd.Series): The data containing anomalies.
        absolute (bool, optional): Activates the use of absolute values. Defaults to False.

    Returns:
        Sequence: An array containing the indexes of the anomalies.
    """
    assert len(data) > 0, "Empty sequence passed."
    _data = np.squeeze(np.array(data))
    if absolute:
        _data = np.abs(_data)
    # assert len(_data.shape) == 1, "This function only works with 1-D data."

    timeSeries = _data[np.isfinite(_data)]
    if timeSeries.ndim == 1:
        timeSeries = timeSeries.reshape(-1, 1)

    clf = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05).fit(timeSeries)
    clf.fit(timeSeries)
    outlier_labels = np.where(clf.predict(timeSeries) == -1, 1, 0)
    anomalies = np.zeros_like(_data)
    anomalies[np.isfinite(_data)] = outlier_labels

    return np.where(anomalies == 1)[0]


def robust_covariance(data: Sequence, absolute: bool = False) -> Sequence:
    """This function identifies anomalies using the Robust covariance algorithm with a suitable set of parameters.

    Uses _Robust covariance_ (Rousseeuw, P.J., Van Driessen, K. “A fast algorithm for the minimum covariance determinant estimator” Technometrics 41(3), 212 (1999)). It works also on multidimensional data.

    Args:
        data (Sequence: list, array, pd.Series): The data containing anomalies.
        absolute (bool, optional): Activates the use of absolute values. Defaults to False.

    Returns:
        Sequence: An array containing the indexes of the anomalies.
    """
    assert len(data) > 0, "Empty sequence passed."
    _data = np.squeeze(np.array(data))
    if absolute:
        _data = np.abs(_data)
    # assert len(_data.shape) == 1, "This function only works with 1-D data."

    timeSeries = _data[np.isfinite(_data)]
    if timeSeries.ndim == 1:
        timeSeries = timeSeries.reshape(-1, 1)

    clf = EllipticEnvelope().fit(timeSeries)
    clf.fit(timeSeries)
    outlier_labels = np.where(clf.predict(timeSeries) == -1, 1, 0)
    anomalies = np.zeros_like(_data)
    anomalies[np.isfinite(_data)] = outlier_labels

    return np.where(anomalies == 1)[0]


def local_outlier_factor(data: Sequence, absolute: bool = False) -> Sequence:
    """This function identifies anomalies using the Local Outlier Factor algorithm with a suitable set of parameters.

    Uses _Local Outlier Factor_ (Breunig, Kriegel, Ng, and Sander (2000) LOF: identifying density-based local outliers. Proc. ACM SIGMOD). It works also on multidimensional data.

    Args:
        data (Sequence: list, array, pd.Series): The data containing anomalies.
        absolute (bool, optional): Activates the use of absolute values. Defaults to False.

    Returns:
        Sequence: An array containing the indexes of the anomalies.
    """
    assert len(data) > 0, "Empty sequence passed."
    _data = np.squeeze(np.array(data))
    if absolute:
        _data = np.abs(_data)
    # assert len(_data.shape) == 1, "This function only works with 1-D data."

    timeSeries = _data[np.isfinite(_data)]
    if timeSeries.ndim == 1:
        timeSeries = timeSeries.reshape(-1, 1)

    pred = LocalOutlierFactor().fit_predict(timeSeries)
    outlier_labels = np.where(pred == -1, 1, 0)
    anomalies = np.zeros_like(_data)
    anomalies[np.isfinite(_data)] = outlier_labels

    return np.where(anomalies == 1)[0]


class scatter_cluster():
    """This class is a tool to access anomaly detection algorithms in an unified and controlled way. 

    It offers methods to help preprocessing data for anomaly detection in time series and allows for a single unified interface to a selection of algorithms.
    The currently available kernels are: IsolationForest, OneClassSVM, RobustCovariance.
    """
    kernels = {"IsolationForest": IsolationForest,
               "OneClassSVM": OneClassSVM, "RobustCovariance": EllipticEnvelope}
    """The kernels available for anomaly detection. It is possible to add your own provided they expose the `fit` and `predict` methods."""
    base_features = ['residual', 'vtot', 'atot', 'angular_velocity']
    """The base features computed during segmentation when no user defined list is passed."""
    _kernels = {"IsolationForest": IsolationForest,
                "OneClassSVM": OneClassSVM, "RobustCovariance": EllipticEnvelope}
    """Convenience attribute to restore `kernels` if it has been modified."""
    _base_features = ['residual', 'vtot', 'atot', 'angular_velocity']
    """Convenience attribute to restore `base_features` if it has been modified."""

    def __init__(self, kernel: str, **kwargs) -> None:
        """Creates a scatter_cluster object with the given kernel.

        Args:
            kernel (str): The name of the kernel used for detection. Must be one of IsolationForest, OneClassSVM, RobustCovariance.
            kwargs: Arguments passed to fine tune the creation of the kernel. These may include random states or other configurable aspects.

        Raises:
            RuntimeError:  In case the kernel specified at creation time is not in the list of the available ones.
        """
        if kernel not in self.kernels:
            raise RuntimeError(
                "The kernel must be any of: "+str(self.kernels))
        self.kernel = self.kernels[kernel](**kwargs)

    def fit(self, data: pd.DataFrame, feature_list: Sequence[str] = None, **kwargs):
        """Fits the kernel for anomaly detection.

        The feature_list allows using a subset of the standard features produced by segment and ensures uniformity with prediction. The special value "all" includes all the columns in the DataFrame with the sole exception of columns named "id", "track_id", "label". Be careful if the DataFrame contains also non relevant columns or columns spuriously correlated with the labels.


        Args:
            data (pd.DataFrame): The output of segment or analogous format with a feature per column. If using custom data specify also the feature_list.
            feature_list (Sequence[str], optional): Lists of features to train the kernel is different from the default one, i.e., all the features from segment. Use "all" for all the columns in the DataFrame. Defaults to None.
            kwargs: Dictionary of arguments for the fit method of the kernel.

        Returns:
            kernel: The fitted estimator.
        """
        if feature_list is None:
            self.feature_list = [
                "_".join([feature, sub]) for feature in self.base_features for sub in ["mean", "max"] if "_".join([feature, sub]) in data]
        elif feature_list == "all":
            self.feature_list = [
                feature for feature in data if feature not in ["id", "track_id", "label"]]
        else:
            self.feature_list = [
                feature for feature in feature_list if feature in data]
        assert len(self.feature_list) > 0
        return self.kernel.fit(data[self.feature_list], **kwargs)

    def predict(self, data: pd.DataFrame, returnLabels: bool = False, **kwargs) -> np.ndarray:
        """Predicts if the points are anomalies or not.

        It uses the same feature list as the fit method.

        Args:
            data (pd.DataFrame): The output of segment or analogous format with a feature per column.
            returnLabels (bool, optional): Instead of returning the indexes of the anomalies, return a vector of labels. Defaults to False.
            kwargs: Dictionary of arguments for the predict method of the kernel.

        Returns:
            np.ndarray: Same len as data, contains the labels for every segment.
        """
        if returnLabels:
            return self.kernel.predict(data[self.feature_list], **kwargs)
        else:
            return np.where(self.kernel.predict(data[self.feature_list], **kwargs) == -1)[0]

    @classmethod
    def segment(cls, data: pd.DataFrame, group_key: str = "track_id", **kwargs) -> pd.DataFrame:
        """Segments a whole dataset at once. See the documentation for `segment_trace`.

        Args:
            data (pd.DataFrame): A dataframe as those produced by `particles2d.preprocess`.
            group_key (str, optional): The column to use for track grouping. Defaults to "track_id".
            kwargs: Arguments for the `segment_trace` function.

        Returns:
            pd.DataFrame: A pd.DataFrame containing the segmented data of all traces, one row per segment.
        """
        if tqdm is not None:
            tqdm.pandas(desc="Segmenting traces")
            return data.groupby(group_key).progress_apply(cls.segment_trace, **kwargs)
        else:
            return data.groupby(group_key).apply(cls.segment_trace, **kwargs)

    @classmethod
    def segment_trace(cls, data: pd.DataFrame, trak_id: Union[Any, None] = None, mwLength: int = 20, step: int = 5, basic: Sequence[str] = [], advanced: Sequence[Tuple[Union[Sequence[str], str], str, Callable[[np.ndarray], float]]] = []) -> pd.DataFrame:
        """Segment one trace for use with the anomaly detection kernel.

        From every segment extracts some relevant statistic for the anomaly detection. The segmentation is carried out with a moving window of size mwLength shifting step points at the time.

        It can compute basic and advanced features. Basic features are the mean and the maximum over the window for selected input features, by default it uses a list of features compatible with the output of `particles2d.preprocess`. Advanced features are defined by the user and can include any function of the values of a single input feature in the window. See example below.

        Args:
            data (pd.DataFrame): A pd.DataFrame as those produced by `particles2d.preprocess`.
            trak_id (Union[Any, None], optional): The id or the grouping parameter of the segmented data. Defaults to None.
            mwLength (int, optional): Length of the moving window for segmentation. Defaults to 20.
            step (int, optional): Sliding step of the window, the lower the step the more and more correlated the segments. Defaults to 5.
            basic (Sequence[str], optional): List of columns for which the basic mean and max values over the window is desired. Defaults to [].
            advanced (Sequence[Tuple[Union[Sequence[str],str], str, Callable[[np.ndarray],float]]], optional): Tuples describing advanced feature extraction. Defaults to [].

        Returns:
            pd.DataFrame: A pd.DataFrame containing the segmented trace.

        Example:

            ```python
            import numpy as np
            from candela.functions import scatter_cluster
            # adding the standard deviation to the computed features:
            advanced_feature = [("my_col", "std", lambda x: np.std(x)),]
            scatter_cluster.segment_trace(data, basic=["my_col"], advanced=advanced_feature)
            ```
        """
        if trak_id is not None:
            t_id = trak_id
        elif hasattr(data, "name"):
            t_id = data.name
        elif "track_id" in data:
            t_id = data["track_id"].unique().tolist()
        else:
            t_id = np.nan

        if len(basic) == 0 and len(advanced) == 0:
            basic = [k for k in cls.base_features if k in data]
            assert len(basic) > 0

        def get_mean(vec):
            return np.mean(vec)

        def get_max(vec):
            return np.max(vec)

        transforms = {}
        for col in basic:
            transforms["_".join([col, "mean"])] = (col, get_mean)
            transforms["_".join([col, "max"])] = (col, get_max)

        for cols, name, func in advanced:
            if isinstance(cols, str):
                cols = [cols]
            for col in cols:
                transforms["_".join([col, name])] = (col, func)

        steps = max(int((len(data) - mwLength)/step), 0)+1
        feature = {k: np.zeros(steps) for k in transforms}
        feature['id'] = np.full(steps, t_id)

        if steps > 0:
            for (i, x) in enumerate(np.arange(0, len(data) - mwLength, step)):
                timeStart = x
                timeEnd = x + mwLength
                t_data = data.iloc[timeStart:timeEnd, :]
                for k in transforms:
                    feature[k][i] = transforms[k][1](t_data[transforms[k][0]])

        out_df = pd.DataFrame(feature)

        return out_df
