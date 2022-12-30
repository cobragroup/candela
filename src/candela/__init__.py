"""The Cobra ANomaly DEtection LibrAry for anomaly detection in timeseries.

The Cobra ANomaly DEtection LibrAry (candela) is a toolbox for the detection and analysis of anomalies and outliers in a wide range of data types.
The toolbox has a modular structure that allows the extension of its functionalities, making it easily adaptable to the needs of various application areas.

It contains several different instruments for the analysis of different types of anomalies in the data, from easy statistical approaches to more advanced machine-learning techniques. The outlier detection techniques based on machine learning in the current version of the software are based on the Scikit-learn library (Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011).

The three submodules of candela are:

- functions: containing the actual functions for anomaly detections.
- plotting (has to be imported separately): containing advanced functions for visualization and reporting of the anomalies.
- particles2D (has to be imported separately): containing functions specialised in working with traces of particles moving on a plane.
"""

from . import functions