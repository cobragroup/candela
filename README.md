# Cobra Anomaly Detection Library

<img align="left" margin-right="30px" src="Logo/candela.png" alt="logo CANDELA" height="73px"></a>The Cobra ANomaly DEtection LibrAry (candela) is a toolbox for the detection and analysis of anomalies and outliers in a wide range of data types, implemented in Python.
Its modular architecture makes it easily adaptable to the needs of various application areas. The toolbox has a modular structure that allows the extension of its functionalities. The outlier detection techniques based on machine learning, which are included in the current version of the software, are based on the Scikit-learn library (Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011).

We present several different instruments for the analysis of different types of anomalies in the data, from easy statistical approaches to more advanced machine-learning techniques. We show the example of use on single features together with the example of useful feature selection, such as the detection of speed and/or angular anomalies in a trajectory. Further, the more complex multivariate feature analysis is presented, which might be beneficial, for example, in the case of too-short trajectories, where usual trajectory detection is problematic. Also, the proposed set of tools might be used for further data analysis, such as the identification of the most informative feature, or distribution of outliers in space and/or time.

Future versions will include further modules, as well as additional parameter options for existing modules. 


## Requirements
Recommended versions: Python >= 3.9.15, scikit-learn >= 1.1.3. Other requirements can be found in [`requirements.txt`](requirements.txt).

## Usage

Download [`requirements.txt`](requirements.txt) the wheel or the dist files from the `dist/` folder, we suggest creating a new conda environment. In case you choose the wheel, on linux we suggest:
``` bash
> conda create -n candela --file requirements.txt
> conda activate candela
> pip install candela-1.0.1-py3-none-any.whl
``` 
For examples of using the toolbox, see the [showcase](showcase.ipynb). Documentation can be found in [`doc.pdf`](doc/doc.pdf).

## Authors

Jaroslav Hlinka, Anna Pidnebesna, Giulio Tani Raffaelli

Institute of Computer Science, Czech Academy of Sciences

COBRA group - http://cobra.cs.cas.cz 

## License

Copyright &copy; 2022, Institute of Computer Science of the Czech Academy of Sciences

This software is made available under the AGPL 3.0 license. For license details see the LICENSE file. For other licensing options including more permissive licenses, please contact the first author (hlinka@cs.cas.cz) or email licensing@cs.cas.cz.


## How to cite

If using this software for academic work, please cite it in your publications as:

Hlinka, J. et al. (2022) COBRA Anomaly Detection Library (CANDELA), GitHub repository, https://github.com/cobragroup/candela.

BibTex:
```
@software{CANDELA, 
author = {Hlinka, Jaroslav and Pidnebesna, Anna and Tani Raffaelli, Giulio}, 
title = {{COBRA Anomaly Detection Library}}, 
year = {2022}, 
url = {https://github.com/cobragroup/candela},
version = {1.0.1}, 
}
```

## Acknowledgment

<a href="https://www.tacr.cz/en/"><img align="left" margin-right="30px" src="Logo/tacr_logo_bw.png" alt="logo TACR" height="73px"></a>
This software was developed within the project TN01000024 National Competence Center ??? Cybernetics and Artificial Intelligence co-financed with the state support of the Technology Agency of the Czech republic within the programme National Centres of Competence 1: Support programme for applied research, experimental development and innovation.
We thanks Pavel Sanda, Filip Blastik, Madhurima Bhattacharjee and Jakub Korenek for algorithm suggestions and testing, and David Hartman for sample data transfer management; and Petr Prasek from CertiCon a.s. for providing with sample trajectory data.

