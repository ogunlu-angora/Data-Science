
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 1-  VERİLERİN IMPORT EDİLMESİ

df=pd.read_csv("././dataset/Credit_risk/credit_risk.csv", sep=",")
df.drop("Unnamed: 0", axis=1, inplace=True)
df["Risk"].replace({"good":0, "bad":1}, inplace=True)
df.head()
    def load_credit():
        df = pd.read_csv("././dataset/Credit_risk/credit_risk.csv", sep=",")
        df.drop("Unnamed: 0", axis=1, inplace=True)
        df["Risk"].replace({"good": 0, "bad": 1}, inplace=True)
        return df
df= load_credit()

# Ön Bakış:

more_cat_cols = [col for col in df.columns if len(df[col].unique()) < 10 and col not in "Risk"]
# Kategorik değişkenleri encode etmek gerekli.
df.isnull().sum() # Eksik değer açısından Caving Account ve Checking Account işlem görmeli
df.describe().T
# Credit Amount scale edilmesi gerekli, min-max aralığı çok yüksek






# 1- Feature Scaling

#For Age
"""df.loc[(df["Age"]<18), "NEW_AGE"]= "Ergen"
df.loc[((df["Age"]>= 18)& (df["Age"]<30)), "NEW_AGE"]="Genc"
df.loc[((df["Age"]>= 30)& (df["Age"]<50)), "NEW_AGE"]="Orta_yas"
df.loc[(df["Age"]>= 50), "NEW_AGE"]="Yasli"
df["NEW_AGE"].nunique()"""

# For Duration / Credit amount

df["Per_year_credit"]= df["Credit amount"]/df["Duration"]
df.head()

df["Per_year_credit"].hist()
plt.show()

df["Per_year_credit"].describe()


# 2- DATA PREP


# A- Transformation : Log dönüşüm yapalım , sağa çarpıklar için


def log_transform(data, selected_data):
    for name in selected_data:
         data[name]= np.log1p(df[name])
selected_data= ["Age","Credit amount","Duration","Per_year_credit"]

df[selected_data].hist()
plt.show()

log_transform(df,selected_data)

# Görselleştirme

plt.figure(dpi=120)
sns.pairplot(df)
plt.show()

#Yorum: Normal dağılıma yakınsamışlar.


# B- MISSING VALUES
#İlk Missing Value değerlendirmesi

def missing_values_table(data):
    cols_with_na = [col for col in df.columns if
                    df[col].isnull().sum() > 0]  # Eksik değerlere sahip değişkenleri yakalamaya yarar
    n_miss = data.isnull().sum().sort_values(ascending=False)  # eksik değerler toplamı azalan şekilde sıralar
    ratio = (data.isnull().sum() / data.shape[0] * 100).sort_values(
        ascending=False)  #eksik değere sahip değişkenlerin eksik değer frekanslarınnyüzzdesel oran
    missing_df = pd.concat([n_miss, np.round(ratio, 3)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)
    return (missing_df)
missing_values_table(df)

# YORUM: Eksik değerleri silmek, bu oranlar ile mantıklı değil. Onun yerine knn uygulayalım. Ama Knn num değerler istediği için dönüşüm yapmak gerekicek.

# ÖNCE Encode işlemi yapılmalı
# ENCODING:

more_cat_cols = [col for col in df.columns if len(df[col].unique()) < 10 and col not in "Risk"]
df.head()


def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=True, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


df, new_cols_ohe = one_hot_encoder(df, more_cat_cols)

df.head()


will_del= [col for col in df.columns if df[col].mean() == 0 ] # Bu değerleri gereksiz üretmiş
will_del
df.drop(columns= will_del, inplace=True)

# 2. missing value değerlendirmesi

# öncelikle Saving_accoutn_nan ve Checking Account Nan değerlerinden 1 olanlara Nan atayalım


nan_replace= ["Saving accounts_nan","Checking account_nan"] # bunlardakiler nan olarak gözüküyor, eksik oalrak algılamıyor.

for i in nan_replace:
    df[i]= df[i].replace(1, np.nan)

df.isnull().sum() # atama sonrası nan değerler geldi.



# knn ile eksik değerlere atama.

#SCALING
### Knn scale işlemi istemektedir.

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
df= pd.DataFrame(scaler.fit_transform(df), columns= df.columns)
df.head()

# KNN Imputer

from sklearn.impute import KNNImputer

imputer= KNNImputer(n_neighbors=5)
df= pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df.isnull().sum()
df.isna().any()


# C- Outlier Değerlendirmesi


def outlier_thresholds(dataframe, variable):
    quartile1=dataframe[variable].quantile(0.25)
    quartile3=dataframe[variable].quantile(0.75)
    interquantile_range=quartile3-quartile1
    up_limit=quartile3+ 1.5*interquantile_range
    low_limit=quartile1-1.5*interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, variable):
    low_limit,up_limit= outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable]<low_limit)|(dataframe[variable]>up_limit)].any(axis=None):
        print(variable, "YES")

for i in df.columns:
    outlier_thresholds(df,i)
    has_outliers(df,i)

numy_cols= [col for col in df.columns if df[col].dtypes !="O"
            and col not in "Risk"
            and col not in new_cols_ohe]
numy_cols
df[numy_cols].boxplot()
plt.show()

# Yorum : Outlier değerler var gözüküyor. Ancak bir de Bunlara LOF ile bakalım bir de

from sklearn.neighbors import LocalOutlierFactor
lof= LocalOutlierFactor(n_neighbors= 20, contamination=0.1)
numy_cols= [col for col in df.columns if df[col].dtypes != "O" and col not in "Risk" and col not in new_cols_ohe]
df1= df[numy_cols]
df.columns
lof.fit_predict(df1)
df1_scores=lof.negative_outlier_factor_
np.sort(df1_scores)[0:40]
""""scores= []
for i in  np.sort(df1_scores):
    score= abs( i - (i+1))
    scores.append(score)
    i+1

from sklearn.neighbors import LocalOutlierFactor
lof= LocalOutlierFactor(n_neighbors= 20, contamination=0.1)
numy_cols= [col for col in df.columns if df[col].dtypes != "O" and col not in "Risk" and col not in new_cols_ohe]
df1= df[numy_cols]
df.columns
lof.fit_predict(df1)
df1_scores=lof.negative_outlier_factor_
np.sort(df1_scores)[0:40]
"""""
scores= []
for i in  np.sort(df1_scores):
    score= abs( i - (i+1))
    scores.append(score)
    i+1

scores

scory= pd.DataFrame(scores)
scory.max()
scory
""""
esik_deger= np.sort(df1_scores)[4]
aykiri_deger= df1_scores > esik_deger


df1[df1_scores==esik_deger] # esik değere gelen değerler

baski_deger= df1[df1_scores==esik_deger]
aykirilar= df1[~aykiri_deger]
res= aykirilar.to_records(index=False)
res[:]=baski_deger.to_records(index=False)
df1[~aykiri_deger]= pd.DataFrame(res, index=df1[~aykiri_deger].index)
df[numy_cols]=df1

scores

scory= pd.DataFrame(scores)
scory.max()
scory""""
esik_deger= np.sort(df1_scores)[4]
aykiri_deger= df1_scores > esik_deger


df1[df1_scores==esik_deger] # esik değere gelen değerler

baski_deger= df1[df1_scores==esik_deger]
aykirilar= df1[~aykiri_deger]
aykirilar
res= aykirilar.to_records(index=False)
res[:]=baski_deger.to_records(index=False)
df1[~aykiri_deger]= pd.DataFrame(res, index=df1[~aykiri_deger].index)
df[numy_cols]=df1

# Veriyi Model İçin Kaydetme

y = df["Risk"]
X = df.drop(["Risk"], axis=1)

X.to_pickle("dataset/Credit_risk/prepared_data/train_all_df.pkl")
y.to_pickle("dataset/Credit_risk/prepared_data/test_all_df.pkl")