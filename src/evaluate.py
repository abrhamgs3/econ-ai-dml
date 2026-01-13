import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate(model, X, y):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    return rmse
