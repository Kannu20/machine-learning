
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


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100, # no. of trees
    max_depth=None,   # let trees grow fully
    random_state=42
)

rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Random Forest",accuracy)

