from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

def train_dml(Y, T, X):
    """
    Double Machine Learning (LinearDML) estimator.
    Y: outcome
    T: treatment
    X: controls
    """
    model_y = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=42
    )

    model_t = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=42
    )

    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=KFold(n_splits=3),
        random_state=42
    )

    dml.fit(Y=Y, T=T, X=X)
    return dml
