from scipy.stats import kendalltau, spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from xgboost import XGBRegressor

xgb_args = {
    "tree_method": "hist",
    "subsample": 0.9,
    "n_estimators": 10000,
    "learning_rate": 0.01
}


def train_and_test(X_train, Y_train, X_test, Y_test, what="accuracy", model_type="XGBRegressor", random_state=None):
    Y1_train = Y_train[what]
    Y1_test = Y_test[what]

    if model_type == "XGBRegressor":
        #        print("Using XGBoost.")
        model = XGBRegressor(**xgb_args, random_state=random_state)
        
    elif model_type == "DTREE":
        # print("Using Random Forest.")
        model = RandomForestRegressor(random_state=random_state)
        # elif model_type == "MLP":
        #     print("Using MLP")
        #     model = MLPRegressor([50, 100, 50], learning_rate_init=0.01, max_iter=1000)
    elif model_type == "Linear":
        #        print("Using Linear Regression")
        model = LinearRegression()
    # elif model_type == "SVM":
    #     print("Using SVM")
    #     model = SVR(kernel="linear")
    else:
        raise ValueError("Unknown model type.")
        
    model.fit(X_train, Y1_train)
    
    Y1_pred = model.predict(X_test)
    score = r2_score(Y1_test, Y1_pred)
    kendal = kendalltau(Y1_test, Y1_pred)[0]

    try:
        feature_importances = {
            col: imp
            for col, imp in zip(X_train.columns, model.feature_importances_)
        }
    except AttributeError:
        try:
            feature_importances = {
                col: imp
                for col, imp in zip(X_train.columns, model.coef_)
            }
        except AttributeError:
            feature_importances =  {}
        
    return score, kendal, feature_importances 
