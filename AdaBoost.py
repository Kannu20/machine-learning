# Staging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns 

df = sns.load_dataset('iris')

print(df.head())

x = df.drop('species', axis = 1)
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

x_train , x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', SVC(probability=True, kernel='rbf', random_state=42)),
    ('lr' , LogisticRegression(max_iter = 1000))
]

meta_learner = LogisticRegression(max_iter=100)

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

stacking_clf.fit(x_train, y_train)

y_pred = stacking_clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)


# AdaBoost Classifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

ada_model = AdaBoostClassifier(
    n_estimators=100,  # Number of weak learners
    learning_rate=1.0,  # Learning rate
    random_state=42
)

ada_model.fit(x_train, y_train)

y_pred_ada = ada_model.predict(x_test)

accuracy_ada = accuracy_score(y_test, y_pred_ada)

print("AdaBoost Classifier Accuracy:", accuracy_ada)

# Gradient Boosting Classifier

gb_model = GradientBoostingClassifier(
    n_estimators=100,  # Number of boosting stages
    learning_rate=0.1,  # Learning rate
    max_depth=3,        # Maximum depth of the individual estimators
    random_state=42
)

gb_model.fit(x_train, y_train)

y_pred_gb = gb_model.predict(x_test)

accuracy_gb = accuracy_score(y_test, y_pred_gb)

print("Gradient Boosting Classifier Accuracy:", accuracy_gb)

# XGBOOST

xgb_model = XGBClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3, use_label_encoder = False, eval_metric = 'mlogloss', random_state=42)

xgb_model.fit(x_train, y_train)

y_pred_xgb = xgb_model.predict(x_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print("XGBOOST Classifier Accuracy:", accuracy_xgb)