import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("data/faturamento_consolidade_sem_feriado.csv", parse_dates=["datetime"])
print("Exibindo df..")
print(df.head())
#dummies_df = pd.get_dummies(df, columns=["feature_05", "feature_01", "feature_02", "feature_11", "feature_12"])
dummies_df = pd.get_dummies(df, columns=["feature_05"])
print(dummies_df.info())

train_df = dummies_df[~dummies_df["datetime"].isin([date(2017,10,1), date(2017,11,1), date(2017,12,1)])]
test_df =  dummies_df[dummies_df["datetime"].isin([date(2017,10,1), date(2017,11,1), date(2017,12,1)])]

str_cols = []
str_cols.append("feature_01")
str_cols.append("feature_02")
str_cols.append("feature_11")
str_cols.append("feature_12")
str_cols.append("datetime")
str_cols.append("receita")
str_cols.append("receita_trim1")
str_cols.append("receita_trim2")
str_cols.append("receita_trim3")
#str_cols.append('categoria_loja')
#str_cols.append('receita_media')
str_cols.append('receita_ano_passado')

X_train = train_df.drop(str_cols, axis=1, errors="ignore")
y_train = train_df["receita"]

X_test = test_df.drop(str_cols, axis=1, errors="ignore")
y_test = test_df["receita"]

print(X_test.columns)
print(X_train.columns)

rf = RandomForestRegressor(n_estimators=500, n_jobs=8, verbose=2)
rf.fit(X_train, y_train)

y_test_pred = rf.predict(X_test)
y_train_pred = rf.predict(X_train) 
print(np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_test_pred)))
