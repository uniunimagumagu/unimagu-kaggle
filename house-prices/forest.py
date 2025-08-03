import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# ✅ データ読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ✅ 欠損値処理（簡易版）
# 数値 → 0で補完、カテゴリ → 'None'で補完
for col in train.columns:
    if train[col].dtype == "object":
        train[col] = train[col].fillna("None")
    else:
        train[col] = train[col].fillna(0)

for col in test.columns:
    if test[col].dtype == "object":
        test[col] = test[col].fillna("None")
    else:
        test[col] = test[col].fillna(0)

# ✅ ラベル
y = train['SalePrice']
train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
test_id = test['Id']
test.drop(['Id'], axis=1, inplace=True)

# ✅ カテゴリ変数をダミー化
all_data = pd.concat([train, test])
all_data = pd.get_dummies(all_data)

# ✅ 再分割
X = all_data.iloc[:len(y), :]
X_test = all_data.iloc[len(y):, :]

# ✅ 学習・検証分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.01, random_state=42)

# ✅ モデル
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ✅ 検証スコア（RMSE）
y_pred_valid = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print(f"Validation RMSE: {rmse:.2f}")

# ✅ テスト予測
predictions = model.predict(X_test)

# ✅ 提出ファイル作成
output = pd.DataFrame({'Id': test_id, 'SalePrice': predictions})
output.to_csv('submission_house_prices.csv', index=False)
print("submission_house_prices.csv saved!")
