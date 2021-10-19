
"""
Değişkenler

Age: Yaş

Sex: Cinsiyet

Job: Meslek-Yetenek (0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)

Housing: Barınma Durumu (own, rent, or free)

Saving accounts: Tasarruf Durumu (little, moderate, quite rich, rich)

Checking account: Vadesiz Hesap (DM - Deutsch Mark)

Credit amount: Kredi Miktarı (DM)

Duration: Süre (month)

Purpose: Amaç (car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

Risk: Risk (Good, Bad Risk)

"""




#  GEREKLILIKLER

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

df.shape

# 2- EDA:

# A- KATEGORİK DEĞİŞKEN ANALİZİ
cat_cols = [col for col in df.columns if df[col].dtypes == 'O' and col not in "Risk" ]

print('Kategorik Değişken Sayısı: ', len(cat_cols))


more_cat_cols = [col for col in df.columns if len(df[col].unique()) < 10 and col not in "Risk"]



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



cat_summary(df, more_cat_cols,"Risk")

# Kategorik GÖrselleştirme:

for col in more_cat_cols:
    sns.countplot(x=col, data=df)
    plt.show()

# Bağımlı Değişkenin Sütun Grafik İle gösterilmesi
sns.countplot(x="Risk", data=df)
plt.show()





# B- SAYISAL DEĞİŞKEN ANALİZİ
df.describe().T

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in "Risk" and col not in "Job"]
print('Sayısal değişken sayısı: ', len(num_cols))

# Yorum: Job değişkeni sayısal gözlem birimleri tarafından ifade edilyor, ama aslında 4 farklı frekansı olan bir kategorik değişken .Bu nedenle
# sayısal değişken olarak dikkate almadım.


# Görselleştirme

#df["Age"].hist()
#plt.show()

# Çaprazlama yapalım

for col in num_cols:
    (sns.
     FacetGrid(df,
               hue="Risk",
               height=5,
              )
     .map(sns.kdeplot,col, shade=True)
     .add_legend())
    plt.show()
# Yorum: Num_cols içindekilerin Risk açısından kırılım bazında durumunu gösterir.

df.shape
df.head()


# Genel Bakış :

plt.figure(dpi=120)
sns.pairplot(df)
plt.show()

# Yorum: Bir önceki yoruma ek olarak, Age ve Credit Amount değişkenlerine log transformation gerekebilir.

# C- TARGET ANALİZİ
df.columns

#  1-Bağımlı değişken açısından
def target_summary_with_cat(data,target):
    cat_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in target]
    for var in cat_names:
        print(pd.DataFrame({"TARGET MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")
target_summary_with_cat(df,"Risk")

# cinsiyet kırılımındaki durum:

df.groupby(["Sex", "Age"]).agg({"Risk": np.mean})

for col in df.columns:
    print(df.groupby(["Sex", col])["Risk"].mean().sort_values(ascending=False))

# Yorum: #Erkekler ortalamada %72 iyi iken, kadınlarda bu oran %65
        # Kadınlarda ortalamada 74 , 53 ve 60 yaşındakilerin risk skor ortalaması sıfır: Veriye bakıp bir tane mi yoksa daha fazla mı gözlem var bu yaş aralığında anlamak gerekir.
        # Job açısından, erkekler her gözlem sınıfında kadınlara göre yaklaşık %10 daha yüksek ortalamaya sahip,
        # Housing açısından: free sınıfında kadınların ortalaması %42 iken erkek ortalaması %63

# 2-   Sayısal Değişkenler Açısından Target Analizi

num_cols

df.groupby("Age")["Credit amount"].mean().sort_values(ascending=False) # yaşlara göre kredi ortalamarı: yaş gözlemi çok sayıda olduğu için anlam çıkarmak kolay değil.
# Ancak en yüksek kredi ortalaması sahip yaş 68 ve 70 yaşları olarak görülüyor.


## Fonksiyon yazalım
def target_summary_with_nums( data, target):
    num_names= [col for col in data.columns if len(data[col].unique())>5
                                                  and data[col].dtypes != "O"
                                                  and col not in target
                                                  "]
    for var in num_names:
        print(data.groupby(target).agg({var: np.mean}), end="\n\n\n")

target_summary_with_nums(df,"Risk")

#Yorum: # Credit Amount açısında: kötü risk kat ( 0 ) ortalaması 3938 iken good risk ( 1 ) kategorisindekilerin 2985
        # Duration açısında: ortalama 24 ay olanların kötü risk, 19 ay olanların iyi risk
        # Age açısından: Kötü risk olanların yaş ortalaması 33 iken, iyi risk sahiplerinin 36.
        # İlk nihai yorum olarak: Kredi miktarı arttıkça, yaş küçüldükçe ve kredi vadesi arttıkça, kötü risk ( bad risk) olma olasılığı artıyor.
        #Ancak bu yorumun istatisksel olarak henüz bir geçerliliği yok. İleri analizler gerekiyor.

#Korelasyona bakalım

df.corr()


def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "Risk":
            pass

        else:
            correlation = dataframe[[col, "Risk"]].corr().loc[col, "Risk"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations

LOW_CORR, HIGH_CORR= find_correlation(df, num_cols)

#Yorum: Age, Credit AMount ve Duration değişkenleri düşük korelasyon değerine sahip çıktı.
#High corelation açısından ise herhangi bir değişken yok .

