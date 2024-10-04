import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from scripts.data_utils import get_connectome
from scripts.classification_models import LogRegPCA



bnu_series_path = 'data/ts_cut/HCPex/bnu{}.npy'
bnu_labels_path = 'data/ts_cut/HCPex/bnu.csv'
ihb_series_path = 'data/ts_cut/HCPex/ihb.npy'
ihb_labels_path = 'data/ts_cut/HCPex/ihb.csv'

X_bnu = np.concatenate([np.load(bnu_series_path.format(i)) for i in (1, 2)], axis=0)
Y_bnu = pd.read_csv(bnu_labels_path)
X_ihb = np.load(ihb_series_path)
Y_ihb = pd.read_csv(ihb_labels_path)

# time series have different length
# by the way ``get_connectome`` reduces them to matrices 419x419

X_bnu = get_connectome(X_bnu)
X_ihb = get_connectome(X_ihb)

# concat the train data
X = np.concatenate([X_bnu, X_ihb])
Y = np.concatenate([Y_bnu, Y_ihb])

# let's split data into train and validation

x_train, x_validate, y_train, y_validate = train_test_split(X, Y,
                                                            test_size=0.15, random_state=10)


logreg = LogRegPCA()
logreg.model.set_params(**{'C': 0.0002})
logreg.pca.set_params(**{'n_components': 0.9})

train_acc = logreg.model_training(x_train, y_train)

# save model and weights 

pkl_filename = "./model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(logreg, file)
