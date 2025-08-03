import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ データ読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ✅ 前処理
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')
embarked_map = {'S': 0, 'C': 1, 'Q': 2}
train['Embarked'] = train['Embarked'].map(embarked_map)
test['Embarked'] = test['Embarked'].map(embarked_map)

train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(train['Age'].median())
test['Fare'] = test['Fare'].fillna(train['Fare'].median())

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features]
y = train['Survived']
X_test = test[features]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ XGBoostモデル（早期終了なし）
model = XGBClassifier(
    n_estimators=200,   # 固定回数
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='error',
    use_label_encoder=False,
    random_state=42
)

# ✅ 学習
model.fit(X_train, y_train)

# ✅ 検証スコア
y_pred_valid = model.predict(X_valid)
print("Validation Accuracy:", accuracy_score(y_valid, y_pred_valid))

# ✅ テスト予測
predictions = model.predict(X_test)

# ✅ 提出ファイル作成
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission_xgb.csv', index=False)
print("submission_xgb.csv saved!")
