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

df= pd.read_csv( "././dataset/Churn/churn.csv", sep=",")
df.head()
df.drop("RowNumber", axis=1, inplace=True)
df.drop("CustomerId", axis=1, inplace=True)
df.drop("Surname",axis=1, inplace=True)
def load_churn():
    df = pd.read_csv("././dataset/Churn/churn.csv", sep=",")
    df.drop("RowNumber", axis=1, inplace=True)
    df.drop("CustomerId", axis=1, inplace=True)
    df.drop("Surname",axis=1, inplace=True)
    return df

#df["Surname"].nunique()

# 2- ÖN BAKIŞ

df.head()

# gizli kategorik değişkenlerde encode yapmaya gerek var mı ?
df.groupby("NumOfProducts")["Exited"].mean()
df.groupby("Tenure")["Exited"].mean()

# Yorum: Tenure açısından sınıfları arasında Exited ortalaması çok farklı değili bu nedenle encode etmeye gerek yok gibi
#   NumOfProducts açısından ise sınıflar arası ortalama farklı, encode etmek işe yarayabilir.



# 3- DATA PREP
# A- Missing Value
df.isna().any() #Eksik değer yok

# Dağılıma bakalım
# B_ DÖNÜŞÜM?
plt.figure(dpi=120)
sns.pairplot(df)
plt.show()

# Yorum: Log ya da üssel dönüşüm ilk etapta gerekli değil gibi duruyor.



# C- FEATURE ENGGINERING

"""df.head()
df["IsPassiveMember"]= df.loc[df["Has"]]"""


df["Balance"].hist()
plt.show()

df.groupby("Balance").agg({"Balance": np.median,"Exited": np.mean})

df["Balance"].value_counts().plot()
plt.show()
#   Balance için bir değişim ya da feature a ihtiyaç var gibi duruyor

# Yorum: Balance değişkeninde 0 değerine sahip 3500  adet değer varkeni diğerleri birbirinden farklı gibi duruyor. Bu durumda Balance için yeni bir feature türetelim.



# For BALANCE

df.loc[df["Balance"]==0, "Balance_ort_not"]= 0
df.loc[df["Balance"]!=0, "Balance_ort_not"]= 1
df.head()



df.corr()  # Korelasyona bakıyoruz, yeni değişkenin değeri fena değil bu veri seti için.


# D- ENCODEr

#more_cat_cols

df.head()

df.groupby("Geography")["Exited"].mean() #Bu değişkeni Encode edilebilir
df.groupby("Tenure")["Exited"].mean() # Çok gerek yok
df.groupby("NumOfProducts")["Exited"].mean() # gerek yok
# df.groupby("NumOfProducts").count()

cat_cols= [col for col in df.columns if df[col].dtypes=="O" ]

df.info()



def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=True)  #dummy_na=True çıkardım çünkü na değer yok #
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


df, new_cols_ohe = one_hot_encoder(df, cat_cols)
df.head()


# E- OUTLIER


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

# Yorum: CreditScore,Age ,Exited , Geography_Spain 'de outlier var gözüküyor. Geography_Spain için dummy değişken yaptık, ayrıca niteliksel bir değişiklik
# olur yapılacak değişim o nedenle kapsamdan çıkarıyorum

#-  Exited, bizim bağımlı değişkenimiz. Dolayısıyla outlier analiz kapsamına almıyorum



# Box plot çizerek bakalım

import seaborn as sns
sns.set_theme(style="whitegrid")

 #Age için
ax = sns.boxplot(x=df["Age"])
plt.show()


lowww, upp=outlier_thresholds(df,"Age")
df[df["Age"]> 62].count() # Box plotta gözüken 60 üstü outlier akpsamında epey kişi var çıkıyor. Ama tabii bu sadece Age açısından bakıldığında,
df[df["Age"]> 64].count()


# CreditScore için
ax = sns.boxplot(x=df["CreditScore"])
plt.show()
 # Yorum: CreditScoreDa ise alt limit açısından outlier değerler gözüküyor
lowww, uppp= outlier_thresholds(df, "CreditScore")

df[df["CreditScore"]< 383].count() # 15 gözlem varmış alt limitten düşük



# LOF ile bakmak daha anlamlı sonuçlar verecektir.


from sklearn.neighbors import LocalOutlierFactor
lof= LocalOutlierFactor(n_neighbors= 20, contamination=0.1)

numy_cols= [col for col in df.columns if df[col].dtypes != "O" if len(df[col].unique()) > 5 and col not in "Exited" and col not in new_cols_ohe and col not in "Tenure"]
df1= df[numy_cols]
df.columns
lof.fit_predict(df1)
df1_scores=lof.negative_outlier_factor_
np.sort(df1_scores)[0:40]

esik_deger= np.sort(df1_scores)[6]
aykiri_deger= df1_scores > esik_deger


df1[df1_scores==esik_deger] # esik değere gelen değerler

baski_deger= df1[df1_scores==esik_deger]
aykirilar= df1[~aykiri_deger]
res= aykirilar.to_records(index=False)
res[:]=baski_deger.to_records(index=False)
df1[~aykiri_deger]= pd.DataFrame(res, index=df1[~aykiri_deger].index)
df[numy_cols]=df1


df.to_pickle("dataset/Churn/prepared_data/prep_churn_not_split.pkl")
df.head()

# Veriyi Model İçin Kaydetme
X = df.drop(["Exited"], axis=1)
y = df["Exited"]
y.head()
X.head()

X.to_pickle("dataset/Churn/prepared_data/independent.pkl")
y.to_pickle("dataset/Churn/prepared_data/dependnt.pkl")

