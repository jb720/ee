# ee

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ee = pd.read_csv('ee export-20180412061012-2.csv')

ee.head()

sns.heatmap(ee.isnull(),yticklabels=False,cbar=False,cmap='viridis')

ee.info()

ee.head(1)

from sklearn.model_selection import train_test_split

X = ee[['total_users','users_who_have_created_reports','intercom_conversations','time_on_essentials','revenue_last_12_mo']]
y = ee['pro_interest']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

columns = list(ee)

data = pd.DataFrame(data=X_test, columns=columns)
data['y_test'] = y_test
data['y_pred'] = logmodel.predict(X_test)

data

data.drop(columns=['cid','name','client_type','implemented','timezone','business_type','primary_customer_type','revenue_model','industry_category','principal_market','opp_stage','pro_interest'])

ee
