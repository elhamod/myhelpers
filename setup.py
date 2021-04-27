from setuptools import setup, find_packages

setup(name='myhelpers', version='1.0', packages=find_packages(where="myhelpers", exclude=["tests"])) # , include=["color.PCA.py","config_plots.py", "ZCA.py", "TrialStatistics.py", "confusion_matrix_plotter.py", "earlystopping.py", "dataset_normalization.py" ] 
