
"""
Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirebilir misiniz?

Amaç bir bankanın müşterilerinin bankayı terk etme ya da terk etmeme durumunun tahmin edilmesidir.

Müşteri terkini tanımlayan olay müşterinin banka hesabını kapatmasıdır.

Veri Seti Hikayesi:

10000 gözlemden ve 12 değişkenden oluşmaktadır.
Bağımsız değişkenler müşterilere ilişkin bilgiler barındırmaktadır.
Bağımlı değişken müşteri terk durumunu ifade etmektedir.
Değişkenler:

Surname : Soy isim
CreditScore : Kredi skoru
Geography : Ülke (Germany/France/Spain)
Gender : Cinsiyet (Female/Male)
Age : Yaş
Tenure : Kaç yıllık müşteri
Balance : Bakiye
NumOfProducts : Kullanılan banka ürünü
HasCrCard : Kredi kartı durumu (0=No,1=Yes)
IsActiveMember : Aktif üyelik durumu (0=No,1=Yes)
EstimatedSalary : Tahmini maaş
Exited : Terk mi değil mi? (0=No,1=Yes)
"""




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
def load_churn():
    df = pd.read_csv("././dataset/Churn/churn.csv", sep=",")
    df.drop("RowNumber", axis=1, inplace=True)
    df.crop("CustomerId", axis=1, inplace=True)
    return df

df.shape

df.info()

df.isna().sum()

# 2- EDA:

# A- KATEGORİK DEĞİŞKEN ANALİZİ

cat_cols= [col for col in df.columns if df[col].dtypes =="O"]
print( "Kategorik değişken sayısı:", len(cat_cols))
cat_cols

# gizli kategorik değişken var mı ?
more_cat_cols= [col for col in df.columns if (len(df[col].unique())<10) and col not in "Exited"]

print( "Gerçek kategorik değişken sayısı: ", len(more_cat_cols))
more_cat_cols
#Yorum:  'IsActiveMember'  ve 'HasCrCard' sayısal ifade edilmiş 1 ve 0 ,
         #"NumOfProducts', sayısal ama encode yapılabilir,"
         #'Exited' ise bizim bağımlı değişkenimiz, 1 ve 0 ifade edilmiş"


def cat_summary(data, categorical_cols, target, number_of_classes=10):
    var_count = 0
    vars_more_classes = []
    for var in categorical_cols:
        if len(data[var].value_counts()) <= number_of_classes:  # sınıf sayısına göre seç
            print(pd.DataFrame({var: data[var].value_counts(),
                                "Ratio": 100 * data[var].value_counts() / len(data),
                                "TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")
            var_count += 1
        else:
            vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)

cat_summary(df,more_cat_cols,"Exited")


# Görselleştirme

#Bağımlı değişkeni
sns.countplot(x="Exited", data=df)
plt.show()

# kategorik değişkenleri
for col in more_cat_cols:
    sns.countplot(x=col, data=df)
    plt.show()


# 3- SAYISAL DEĞİŞKEN ANALİZİ
df.describe().T
num_cols= [col for col in df.columns if df[col].dtypes!= "O" and col not in more_cat_cols and col not in "Exited"]

df.corr()
# Yorum: Bağımlı değişken açısından Age, Balance ve EstimatedSalary etki olarak daha yüksek gibi duruyor



# Görselleştirme

# Genel Açıdan :

plt.figure(dpi=120)
sns.pairplot(df[num_cols])
plt.show()

#Çaprazlama
for col in num_cols:
    (sns.
     FacetGrid(df,
               hue="Exited",
               height=5,
              )
     .map(sns.kdeplot,col, shade=True)
     .add_legend())
    plt.show()

#Yorum# Age değişkeninde bağımlı değişken sınıfları açısından fark var gibi,
        # Tenure, yani müşterinin kaç yıllık olduğunda pek fark yok
        # CustomerId zaten unique o nedenle çok önemli değil
        # Credit Score açısından da çok farklılık yok gibi
#df["CustomerId"].nunique() #görüleceği üzere CustomerId unique değerler. O nedenle analizden çıkarılabilir.

# 4- TARGET ANALİZİ

# Genel Bakış:
def target_summary_with_cat(data,target):
    cat_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in target]
    for var in cat_names:
        print(pd.DataFrame({"TARGET MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")

target_summary_with_cat(df,"Exited")

#Yorum: Doğrudan:
#       HasCrCard değişkenin Exited açısından ortalaması aynı , çok da bir etkisi yok gibi duruyor
#       IsActiveMember için ortalama farklı gibi duruyor, üzerine düşünmek iyi olur
#       Cinsiyet açısından da farklı duruyor

df.pivot_table(df, columns="Exited") # mean() açısından bakalım


# Kırılım Bazında

df.groupby(["Gender","IsActiveMember","Age"]).agg({"Exited": np.mean})
df.columns


pd.pivot_table(df, values=["Exited"], index= ["Gender","HasCrCard","IsActiveMember"], aggfunc={"Exited": np.mean})
pd.pivot_table(df, values=["Exited"], index= ["Gender","IsActiveMember"], aggfunc={"Exited": np.mean})

# Yorum: Kredi kartına sahip olup aktif üyeliği 1 ve 0 olanlar var ve Target açısından ortalamaları farklı. Bir hidden pattern olabilir.
#       Bu durumda Isactive Member olup kredi kartı olan ve olmayan yerine akrt varsa aktif, yoksa pasif üye diye yeni bir değer yapılabilir.

pd.pivot_table(df, values=["Exited","CreditScore","Balance"],
               index= ["Gender","HasCrCard","IsActiveMember"],
               aggfunc={"CreditScore": np.mean,
                        "Exited": np.mean,
                        "Balance": np.median})
# Yorum:  Kadın olup, kredi kartı olmayan ve aktiv uye olmayanların %30 u çıkıyor, aynı durumdaki erkeklerde oranı %21 civarı
#         Kadın olup, kredi kartı olan ama aktif üye olmayanların %32 çıkıyor, aynı durumdaki erkeklerde bu oran %22. Cinsiyet durumu etkiliyor oalbilir.
#         Kadın olup, kredi kartı olsun olmasın ama aktif üye ise oranlar aktif üye olmayanlara göre daha düşük . Demek ki aktif üyelik etkiliyor olabili.r

#Fonksiyon

def target_summary_with_nums( data, target):
    num_names= [col for col in data.columns if len(data[col].unique())>5
                                                  and data[col].dtypes != "O"
                                                  and col not in target
                                                  ]
    for var in num_names:
        print(data.groupby(target).agg({var: np.mean}), end="\n\n\n")

target_summary_with_nums(df,"Exited")


#Yorum : Çıkan ve çıkmayanların Balance(Bakiye) ortalaması farklı, demek  ki bakiye bir gösterde olabilir.
#       Age açısından bakarsak, genç olanlar çıkma eğiliminde gibi
#       Kredi skoru açısından da düşük puan ortalamasına sahip çıkanlar



#Korelasyon analizi


def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "Exited":
            pass

        else:
            correlation = dataframe[[col, "Exited"]].corr().loc[col, "Exited"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations

LOW_CORR, HIGH_CORR= find_correlation(df, num_cols)


LOW_CORR # Düşük korelasyona sahip olanlar arasından modellerde düşük önem düzeyine sahip olanları çıkarabiliriz. Ama unutmamak gerekir ki:
# basit modellerde korelasyon bir göstergeyken, birikimli modellerde düşük korelasyona sahip değişkenler de yüksek etkiye sahip çıkabilir.