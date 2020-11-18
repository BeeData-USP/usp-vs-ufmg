#import xgboost as xgb
import pandas as pd
from datetime import date
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split

df = pd.read_csv("data/dados_desafio_2.csv")

grid = {'learning_rate': [0.03, 0.05, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'n_estimators': [100, 250, 500],
        #'max_leaves': [i for i in range(2, 10)]
}

X = df.drop("receita", axis=1)
y = df["receita"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(grid)
cb_model = CatBoostRegressor(iterations=1500,
                             learning_rate=0.05,
                             depth=10,
                             eval_metric='RMSE',
                             random_seed = 42,
                             thread_count=16,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)

grid_search_result = cb_model.grid_search(grid, 
                                       X=X_train, 
                                       y=y_train, 
                                       plot=True)

cb_model.save_model("modelo_desafio_2",
           format="cbm",
           export_parameters=None,
           pool=None)

print(grid_search_result)
y_pred = cb_model.predict(X_test) 
print(np.sqrt(train_test_split(y_test, y_pred)))


