# create script, which loads model, does all preprocessing and outputs solution.csv

import numpy as np
import pandas as pd
import pickle
from scripts.data_utils import get_connectome
from scripts.classification_models import LogRegPCA

X = np.load('data/ts_cut/HCPex/predict.npy')
X = get_connectome(X)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

y_pred = model.model_predict(X)

solution = pd.DataFrame(data=y_pred, columns=['prediction'])
solution.to_csv('./solution.csv', index=False)
