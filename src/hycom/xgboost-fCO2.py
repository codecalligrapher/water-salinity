import cudf
import cupy as cp
import plotly.graph_objects as go
import datashader as ds
import colorcet
import os
from statsmodels.regression import linear_model
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from cuml.preprocessing.model_selection import train_test_split

from tqdm import tqdm


import graphviz

# model analysis
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import mean_absolute_error

import xgboost as xgb


# Visualisation Imports
import numpy as np
import xarray as xr
# datashader
import datashader as ds
import datashader.transfer_functions as tf
from datashader.transfer_functions import shade
from datashader.transfer_functions import stack
from datashader.transfer_functions import dynspread
from datashader.transfer_functions import set_background
from datashader.transfer_functions import Images, Image
from datashader.colors import Elevation
from datashader.utils import orient_array

# holoviews
import holoviews as hv
from holoviews.plotting.plotly.dash import to_dash
from holoviews.element.tiles import CartoDark
from holoviews.operation.datashader import datashade, shade, dynspread, spread, rasterize
from holoviews.operation import decimate

# plotly
from plotly.colors import sequential
from plotly.subplots import make_subplots

# Dash Import
import dash
import dash_html_components as html
from jupyter_dash import JupyterDash

# XGBoost
import xgboost as xgb

DATA_PATH = '../../data/hycom'
RES_PATH = '../../results/hycom'

X_col_names = ['water_temp_0', 'salinity_0', 'water_temp_2', 'salinity_2', 'water_temp_4', 'salinity_4', 'water_temp_6', 'salinity_6', 'water_temp_8', 'salinity_8']

y_col_name = ['fCO2_SW@SST_uatm']

df = cudf.read_csv(os.path.join(RES_PATH, 'hycom_equinox_merged-201920.csv'))
df = df[df['WOCE_QC_FLAG'] == 2]
df.drop(['start_date', 'lat', 'lon', 'WOCE_QC_FLAG', 'easting', 'northing'], axis=1, inplace=True)

df = df.dropna(subset=X_col_names + y_col_name, axis=0)

df = df[df[y_col_name[0]] >= 0]

X, y = df[X_col_names], df[y_col_name]

poly_features = PolynomialFeatures(2, interaction_only=True, include_bias=False) 
X = poly_features.fit_transform(X.as_gpu_matrix())
X = cudf.DataFrame(X, columns=poly_features.get_feature_names(X_col_names))

X_train, X_test, y_train, y_test = train_test_split(X, y)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)


params = {
    'max_depth':    4,
    'max_leaves':   2**8,
    'tree_method':  'gpu_hist',
    'objective':    'reg:squarederror',
    'eval_metric': 'mphe'
}


min_error = float("Inf")
best_params = None

grid_search_params = [
    (max_depth, min_child_weight, subsample, eta)
    for max_depth in range(3,8) 
    for min_child_weight in range(5,7)
    for subsample in [i/10 for i in range(7, 10)]
    for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]
]




grid_search_params = tqdm(grid_search_params)

with open('logs.txt', 'a') as file:
    file.write('max_depth,min_child_weight,eta,subsample,boost_rounds,mean_error')
    for max_depth,min_child_weight,subsample,eta in grid_search_params:
        params['max_depth'] = max_depth
        params['subsample'] = subsample
        params['eta'] = eta
        params['min_child_weight'] = min_child_weight

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=999,
            seed=42,
            nfold=5,
            early_stopping_rounds=10
        )

        mean_error = cv_results['test-mphe-mean'].min()
        boost_rounds = cv_results['test-mphe-mean'].argmin()


        grid_search_params.set_description(f'\tmphe {mean_error} for {boost_rounds} rounds, eta: {eta}, max_depth: {max_depth}')

        if mean_error < min_error:
            min_error = mean_error
            best_params = (max_depth,min_child_weight,subsample,eta)

        file.write(f'{max_depth},{min_child_weight},{eta},{subsample},{boost_rounds},{mean_error}\n')


