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

# VERININ HAZIRLANMASI

# Kaydedilmiş Verinin Çağırılması
X= pd.read_pickle("././dataset/Credit_risk/prepared_data/train_all_df.pkl")
y = pd.read_pickle("././dataset/Credit_risk/prepared_data/test_all_df.pkl")
y.head()
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
                                index=X.columns).sort_values(ascending=False)
    
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title("Değişken Önem Düzeyleri")
        plt.show()
        df.head()"""

#boxplot algorithm comparison
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


# 2- TUM MODELLER CV YONTEMI

models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('SVM', SVC(gamma='auto')),
          ('XGB', GradientBoostingClassifier()),
          ("LightGBM", LGBMClassifier())]

#evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123456)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison # accuracy skor dağılımları ( mean olmayan halleri)
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Yorum: CrossValidation yönteminde sonuçlar daha düşük geldi. O nedenle holdout ile devam edilmesine karar verdim.

# Holdout yüksek gelen LightGBM model'i tune ederek devam edelim



# 1- LightGBM : Holdout + CV ile

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
cross_val_score(lgbm_tuned, X_train, y_train, cv=10).mean() #modelin 10 sefer çalıştıktan sonraki ortalama accuracy değerini göstericek.

feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()
#df.head()



# bu modeli test verisinde deneyelim
y_pred = lgbm_tuned.predict(X_test)
acc = accuracy_score(y_test, y_pred)
msg = "%s: (%f)" % (name, acc)
print(msg)
print(confusion_matrix(y_test, y_pred))
print( classification_report(y_test, y_pred))

import pickle
pickle.dump(lgbm_tuned, open('lgbm_tuned.pkl','wb'))

# 3 CART
from sklearn.tree import DecisionTreeClassifier
from skompiler import skompile
cart= DecisionTreeClassifier()
cart_model= cart.fit(X_train,y_train)
print( skompile(cart_model.predict)).to( "phyton/code")
df.columns
# test verisi üzerindeki skoru
y_pred= cart_model.predict(X_test)
accuracy_score(y_test, y_pred)

"""
#2-
log_model = LogisticRegression().fit(X_train, y_train)
log_model.intercept_
log_model.coef_

log_model.predict(X)[0:10] # bir threshold a göre olasılıkları dönüştürerek getirir
y[0:10]

log_model.predict_proba(X)[0:10] # OLASILIKLARI TAHMİN ET, # Sol taraf "0", sağ taraf "1" gelme olasılığı,
# eğer threshold 0.5 ise ilk tahmine sağ taraf büyük olduğu için 1 gelir.
# SKOR:
# 1- Train verisi üzerinden skor
kfold = KFold(n_splits=10, random_state=123456)
cross_val_score(log_model, X_train, y_train, cv=kfold).mean() # her biri için 9 parça model,biri ile test et

##NOT: Hold-out yöntemine göre cross_val daha doğru sonuç verir, verinin farklı noktalarına baktığı için.
# #Daha iyi kastedilmiyor, daha doğru diyoruz.
# 2- test verisi üzerinde skor
y_pred= log_model.predict(X_test)
accuracy_score(y_test,y_pred)
Y_pred_new= (log_model.predict_proba(X_test)[:,1]>= 0.9).astype(bool)
print(classification_report(y_test, y_pred))
print(classification_report(y_test, Y_pred_new))
accuracy_score(y_test,Y_pred_new)

print( confusion_matrix(y_test, y_pred))
print( confusion_matrix(y_test, Y_pred_new))

"""





#NOT: Aşağıdaki işlemleri yapınca sonuç 0.90 'dan 0.70 lere düşüyor. Demek ki , hidden-pattern var
# önem düzeyi düşük çıkan değişkenler açısından, bu olabilir. Ayrıca bu değişkenleri çıkarmak için, korelasyon değerlerine de bakmakta fayda var Eğer koralasyon skorları düşükse çıkarılabilir.




"""
# önem düzeyi zayıf değişkenleri çıkarıp modeli

dlet_col= ["Saving accounts_nan","Purpose_vacation/others","Purpose_repairs","Purpose_domestic appliances","Checking account_nan",#"NEW_AGE_Yasli",
           "Checking account_rich"]

for i in dlet_col:
    print ( i in X)

X= X.drop(columns=dlet_col) #önem değeri düşük olanları silip tekrar split edelim

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

# Tekrar Model Kuralım: LGBM Hold_out + CV

lgbm = LGBMClassifier(random_state=12345)
cross_val_score(lgbm, X_train, y_train, cv=10).mean()

# model tuning
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
               "n_estimators": [500, 1000, 1500],
               "max_depth": [3, 5, 8]}

gs_cv = GridSearchCV(lgbm,
                     lgbm_params,
                     cv=5,
                     n_jobs=-1,
                     verbose=2).fit(X_train, y_train)
# train üzerinde test et
lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X_train, y_train)
cross_val_score(lgbm_tuned, X_train, y_train, cv=10).mean()

#Test verisi üezerinde
y_pred = lgbm_tuned.predict(X_test)
acc = accuracy_score(y_test, y_pred)
msg = "%s: (%f)" % (name, acc)
print(msg)
print(confusion_matrix(y_test, y_pred))

"""





