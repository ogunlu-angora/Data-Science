
# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None)

# GET DATA FROM PICKLE FILE

df = pd.read_pickle(r'dataset/Churn/prepared_data/prep_churn_not_split.pkl')


#  SCORRING (1-10)

# Dataset
df1= df.drop("New_Exited", axis=1)
df1.head()

## Target and features
X= df1.drop("Exited", axis=1)
Y= df1["Exited"]

## Split
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.20,
                                                    random_state=42)

## Model
lgbm_model = LGBMClassifier().fit(X_train, y_train)

y_pred = lgbm_model.predict(X_test)
accuracy_score(y_test, y_pred)

## Proba function
print( lgbm_model.predict_proba(X_test))


probability= lgbm_model.predict_proba(X_test)[:,1]

# Assign the proba.. value for the churn predicted
"""

df1_new['prob_0'] = prediction_of_probability[:,0] 
df1_new['prob_1'] = prediction_of_probability[:,1]"""

indices= X_test.index.values

df1_new= df1.copy()

df1_new.loc[indices, "proba_tes"]=probability

df1_new.head()


## Cutting by the cathegory

df1_new["cathegory"]= pd.cut(df1_new["proba_tes"], 10, labels=[1,2,3,4,5,6,7,8,9,10])

df1_new["cathegory"].nunique()



df1_new.sort_values("cathegory", ascending= False).head(50)
df1_new.sort_values("cathegory", ascending= False).tail(50)



# RFM ( random score)

df1_new["R"]= np.random.randint(1,6, df1_new.shape[0])
df1_new["F"]= np.random.randint(1,6, df1_new.shape[0])

df1_new["RF_score"]= df1_new["R"].astype(str)+ df1_new["F"].astype(str)

df1_new[df1_new["RF_score"]=="55"].head()



## Segmentation

seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Loose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}
## Assign the RF_score to Segmentation

df1_new["Segment"]= df1_new["RF_score"]
df1_new.head()

df1_new["Segment"]=df1_new["Segment"].replace(seg_map, regex=True)

df1_new.head()


# Last Comment:

"""who are in the Champions , and at the same time their Churn cathegories ( churn score) are between 7-10"""



df1_new[df1_new.Segment=="Champions"]
df1_new[df1_new.cathegory >= 7]


df1_new[(df1_new.Segment =="Champions")& (df1_new.cathegory >= 7)].head()