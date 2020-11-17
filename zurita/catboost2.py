#import xgboost as xgb
import pandas as pd
from datetime import date
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("data/faturamento_consolidade_com_feriado_e_receita_da_categoria.csv", parse_dates=["datetime"])

grid = {'learning_rate': [0.03, 0.05, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'n_estimators': [100, 250, 500],
        #'max_leaves': [i for i in range(2, 10)]
}

print("Exibindo df..")
print(df.head())
#dummies_df = pd.get_dummies(df, columns=["feature_05", "feature_01", "feature_02", "feature_11", "feature_12"])
dummies_df = pd.get_dummies(df, columns=["feature_05"])
print(dummies_df.info())

train_df = dummies_df[~dummies_df["datetime"].isin([date(2017,10,1), date(2017,11,1), date(2017,12,1)])]
test_df =  dummies_df[dummies_df["datetime"].isin([date(2017,10,1), date(2017,11,1), date(2017,12,1)])]

str_cols = []
str_cols.append("datetime")
str_cols.append("receita")

str_cols.append("feature_01")
str_cols.append("feature_02")
str_cols.append("feature_11")
str_cols.append("feature_12")
str_cols.append("receita_trim1")
str_cols.append("receita_trim2")
str_cols.append("receita_trim3")
str_cols.append('receita_ano_passado')
str_cols.append('cod_loja')
#str_cols.append('categoria_loja')
#str_cols.append('receita_media')

X_train = train_df.drop(str_cols, axis=1, errors="ignore")
y_train = train_df["receita"]

X_test = test_df.drop(str_cols, axis=1, errors="ignore")
y_test = test_df["receita"]



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

# cb_model.fit(X_train, y_train,
#              eval_set=(X_test, y_test),
#              use_best_model=True,
#              verbose=50)

grid_search_result = cb_model.grid_search(grid, 
                                       X=X_train, 
                                       y=y_train, 
                                       plot=True, cv=TimeSeriesSplit())

cb_model.save_model("modelo_catboost_otimizado",
           format="cbm",
           export_parameters=None,
           pool=None)

print(grid_search_result)
