import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

pd.pandas.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore")

#VERININ HAZIRLANMASI

# Kaydedilmiş Verinin Çağırılması
X= pd.read_pickle("dataset/Churn/prepared_data/independent.pkl")
X.head()
y = pd.read_pickle("dataset/Churn/prepared_data/dependnt.pkl")
y.head()
X.head()
#Model:
# 1- HOLDOUT + CV ( ikisi bir arada, önce train-test olarak böl, train'e cross_val yap)
# train-test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('SVM', SVC(gamma='auto')),
          ('XGB', GradientBoostingClassifier()),
          ("LightGBM", LGBMClassifier())]

# evaluate each model in turn
results = []
names = []



for name, model in models:
    kfold = KFold(n_splits=10, random_state=123456)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    """if (model.feature_importances_) :
        feature_imp = pd.Series(model.feature_importances_,
        feature_imp = pd.Series(model.feature_importances_,
                                index=X.columns).sort_values(ascending=False)

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title("Değişken Önem Düzeyleri")
        plt.show()
        df.head()"""

# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# test verisi üzerinden accuracy ve confusion matrix  bakalım
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    msg = "%s: (%f)" % (name, acc)
    print(msg)
    print(confusion_matrix(y_test, y_pred))



"""# 2- RF

rf_model = RandomForestClassifier(random_state=12345).fit(X_train, y_train)
cross_val_score(rf_model, X_train,y_train,cv=10).mean()

#Tuning RF
rf_params = {"n_estimators": [200, 500],
             "max_features": [5, 7],
             "min_samples_split": [5, 10],
             "max_depth": [5, None]}


rf_model = RandomForestClassifier(random_state=12345)

gs_cv = GridSearchCV(rf_model,
                     rf_params,
                     cv=10,
                     n_jobs=-1,
                     verbose=2).fit(X_train, y_train)

gs_cv.best_params_

rf_tuned = RandomForestClassifier(**gs_cv.best_params_)
cross_val_score(rf_tuned, X_train, y_train, cv=10).mean()

Importance= pd.DataFrame({"Importance" : rf_model.feature_importances_*100},
                         index=X_train.columns)

# bu modeli test verisinde deneyelim
y_pred = rf_tuned.predict(X_test)
acc = accuracy_score(y_test, y_pred)
msg = "%s: (%f)" % (name, acc)
print(msg)
print(confusion_matrix(y_test, y_pred))
print( classification_report(y_test, y_pred))

#RF Nerden bölmüş verileri
from sklearn.tree import DecisionTreeClassifier
from skompiler import skompile


print( skompile(rf_tuned.predict)).to( "phyton/code")
"""



# 3- LightGBM
lgbm = LGBMClassifier(random_state=12345)
cross_val_score(lgbm, X_train, y_train, cv=10).mean()
# Tuning
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
               "n_estimators": [500, 1000, 1500],
               "max_depth": [3, 5, 8]}

gs_cv = GridSearchCV(lgbm,
                     lgbm_params,
                     cv=5,
                     n_jobs=-1,
                     verbose=2).fit(X, y)

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X, y)
cross_val_score(lgbm_tuned, X_train, y_train, cv=10).mean()

# importance
feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()


# bu modeli test verisinde deneyelim
y_pred = lgbm_tuned.predict(X_test)
acc = accuracy_score(y_test, y_pred)
msg = "%s: (%f)" % (name, acc)
print(msg)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


