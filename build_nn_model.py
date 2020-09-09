import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network.multilayer_perceptron import MLPRegressor

from transformers.pca import GenesisPCA
from transformers.era_transformers import EraScaler, EraNumberer, EraToNumber, EraDropper
from transformers.col_droppers import ColumnDropper
from transformers.cluster import EraClusterer

hard_dropping = False
try:
    feature_picker = pd.read_pickle("./data/nmr_picker.pkl")
except:
    pass


def build_nn_model(params):
    if params is None:
        params = {
            'scaler': StandardScaler(),
            'era_scaler': False,
            'era_numberer': False,
            'pca_var_limit': 1.0,
            'pca_minor_components': 0,
            'feature_picker': False,

            'hidden_layer_1_size': 20,
            'hidden_layer_2_size': 0,
            'hidden_layer_3_size': 0,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 200,
            'learning_rate_init': .001,
            'max_iter': 200,
            'shuffle': True,
            'early_stopping': False,
            'validation_fraction': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'n_iter_no_change': 10,
            'n_clusters': 0
        }
        if hard_dropping:
            for i in range(311):
                params[f"{i}"] = False

    ppl_steps = []

    if params["feature_picker"]:
        ppl_steps.append(('feature_picker', feature_picker))
    # ppl_steps.append(("EraToNumber", EraToNumber()))

    if hard_dropping:
        # col dropping logic
        col_indexes = []
        for i in range(311):
            col_indexes.append(params[f"{i}"])
        ppl_steps.append(('col_dropper', ColumnDropper(col_indexes=col_indexes)))

    if params["n_clusters"] > 1:
        ppl_steps.append(('era_clusterer', EraClusterer(n_clusters=params["n_clusters"])))
    if params['era_numberer']:  # this has to come before scaling so that it is scaled before we get to pca
        ppl_steps.append(('era_numberer', EraNumberer()))
    if params['era_scaler']:
        ppl_steps.append(('era_scaler', EraScaler()))
    ppl_steps.append(('era_dropper', EraDropper()))
    if params['scaler']:
        ppl_steps.append(('scaler', params['scaler']))
    if params['pca_var_limit']:
        ppl_steps.append(('pca', GenesisPCA(var_limit=params["pca_var_limit"],
                                            minor_components=params["pca_minor_components"])))

    # get the hidden layer tuple
    l = [params["hidden_layer_1_size"], params["hidden_layer_2_size"], params["hidden_layer_3_size"]]
    hidden_layer_sizes = tuple([int(i) for i in l if i > 0])

    # now make mlp model
    nn_model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=params['activation'],
        solver=params['solver'],
        alpha=params['alpha'],
        batch_size=params['batch_size'],
        learning_rate_init=params['learning_rate_init'],
        max_iter=params['max_iter'],
        shuffle=params['shuffle'],
        early_stopping=params['early_stopping'],
        validation_fraction=params['validation_fraction'],
        beta_1=params['beta_1'],
        beta_2=params['beta_2'],
        n_iter_no_change=params['n_iter_no_change']
    )
    ppl_steps.append(('mlp', nn_model))

    return Pipeline(ppl_steps)


def get_possible_params():
    possible_params = {
        'scaler': [StandardScaler()],
        'era_scaler': [False, True],
        'era_numberer': [False],
        'pca_var_limit': [i for i in np.geomspace(0.2, 1.0, 20)] + [False],
        'pca_minor_components': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6],
        'feature_picker': [False],

        'hidden_layer_1_size': [int(i) for i in np.geomspace(10, 200, 100)],
        'hidden_layer_2_size': [int(i) for i in np.geomspace(10, 200, 100)] + [0]*100,
        'hidden_layer_3_size': [int(i) for i in np.geomspace(10, 200, 100)] + [0]*200,
        'activation': ['identity', 'logistic', 'tanh', 'relu', 'relu', 'relu', 'identity'],
        'solver': ['adam'],
        'alpha': [0.00001, 0.00005, 0.0001, 0.0005, 0.001],
        'batch_size': [5000, 1000, 100, 10],
        'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'max_iter': [200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000],
        'shuffle': [True, True, False],
        'momentum': [0.9],
        'early_stopping': [True, False],
        'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5],
        'beta_1': [0.8, 0.85, 0.9, 0.95, 0.99, 0.995, 0.999],
        'beta_2': [0.8, 0.9, 0.99, 0.999, 0.9999],
        'n_iter_no_change': [50, 500],
        'n_clusters': [0]
    }
    if hard_dropping:
        for i in range(311):
            possible_params[f"{i}"] = [False]*9 + [True]*1
    return possible_params


def get_minor_mutations():
    minor_mutations = {
        'era_scaler': lambda x: (not x) if np.random.uniform(0, 1) > 0.8 else x,
        'era_numberer': lambda x: (not x) if np.random.uniform(0, 1) > 0.8 else x,
        'pca_var_limit': lambda x: np.clip(x * np.random.uniform(0.8, 1.2) if x else np.random.uniform(0.9, 0.99), 0.1, 1.0),
        'pca_minor_components': lambda x: np.clip(x + np.random.choice([-1, 1]), 0, 6),
        # 'feature_picker': lambda x: (not x) if np.random.uniform(0, 1) > 0.9 else x,

        'hidden_layer_1_size': lambda x: int(np.clip(x*np.random.uniform(0.8, 1.2), 10, 500)),
        'hidden_layer_2_size': lambda x: int(np.clip(x * np.random.uniform(0.8, 1.2), 10, 500)),
        'hidden_layer_3_size': lambda x: int(np.clip(x * np.random.uniform(0.8, 1.2), 10, 500)),
        'activation': lambda x: np.random.choice(['identity', 'tanh', 'logistic', 'relu']) if np.random.uniform(0, 1)>0.8 else x,
        'solver': lambda x: x,
        'alpha': lambda x: np.clip(x * np.random.uniform(0.8, 1.2), 0.000001, 0.01),
        'batch_size': lambda x: int(np.clip(x * np.random.uniform(0.8, 1.2), 1, 50000)),
        'learning_rate_init': lambda x: np.clip(x * np.random.uniform(0.8, 1.2), 0.00001, 0.01),
        'max_iter': lambda x: int(np.clip(x*np.random.uniform(0.8, 1.2), 100, 5000)),
        'shuffle': lambda x: not x if np.random.uniform(0,1)>0.8 else x,
        'early_stopping': lambda x: not x if np.random.uniform(0,1)>0.8 else x,
        'validation_fraction': lambda x: np.clip(x * np.random.uniform(0.8, 1.2), 0.01, 0.8),
        'beta_1': lambda x: x,
        'beta_2': lambda x: x,
        'n_iter_no_change': lambda x: np.clip(x+np.random.choice([-1, 1]), 1, 200),
        # 'n_clusters': lambda x: np.clip(x + np.random.choice([-1, 1]), 0, 10),
    }
    if hard_dropping:
        for i in range(311):
            minor_mutations[f"{i}"] = lambda x: (not x) if np.random.uniform(0, 1) > 0.95 else x

    return minor_mutations
