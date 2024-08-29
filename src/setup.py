# src/setup.py

from setuptools import setup, find_packages

setup(
    name='spacex_falcon9_landing_prediction',
    version='1.0.0',
    author='Mourad ouzziki',
    author_email='mouradouzziki@gmail.com',
    description='Predicting successful landings of SpaceX Falcon 9 first stage',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'plotly',
        'xgboost',
        'joblib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
