#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, r2_score, precision_score, recall_score, f1_score


# In[2]:


df = pd.read_csv(r"C:\Users\KHUSHI\Downloads\archive (5)\Churn_Modelling.csv", header = 0)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().values.any()


# In[9]:


df.columns


# In[10]:


df.index


# In[11]:


print(df["Geography"].unique())
print(df["Gender"].unique())
print(df["NumOfProducts"].unique())
print(df["HasCrCard"].unique())
print(df["IsActiveMember"].unique())
print(df["Exited"].unique())


# In[12]:


df.isnull().sum()


# In[13]:


df.axes


# In[14]:


df.iloc[0:4]


# In[15]:


df.loc[:, "CustomerId"]


# In[16]:


sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.show()


# In[17]:


plt.figure(figsize=(14,10))
sns.countplot(x='Geography', data=df)
plt.xlabel('Geography Distribution')
plt.ylabel('Count')
plt.title('Geography Distribution Plot',fontsize=14, fontweight="bold", color = "purple")
plt.show()


# In[18]:


sns.displot(df.Geography, kde =True, color = "red")


# In[19]:


sns.displot(df.Geography, color = "yellow")


# In[20]:


plt.figure(figsize = (14,10))
count1 = df.Geography.value_counts()
sns.barplot(x = count1, y = count1.index, orient = 'h')
plt.xlabel('Count')
plt.ylabel('Geography')
plt.title('Geography Distribution Plot')
plt.xticks(rotation=90)
plt.show()


# In[21]:


plt.figure(figsize = (14,10))
count1 = df.Geography.value_counts()
sns.barplot(x = count1, y = count1.index, orient = 'h')
plt.xlabel('Count', fontsize = 16, fontweight = 'bold')
plt.ylabel('Geography', fontsize = 16, fontweight = 'bold')
plt.title('Geography Distribution Plot', fontsize = 24, fontweight = 'bold', color = 'blue')
plt.xticks(rotation=90, fontsize = 12, fontweight = 'bold', color = 'green')
plt.yticks(fontsize = 12, fontweight = 'bold', color = 'green')
plt.show()


# In[52]:


correlation_matrix = df.corr()
plt.figure(figsize = (12,8))
sns.heatmap(correlation_matrix, cmap = 'coolwarm', annot = False,  fmt=".2f")
plt.title("Correlation Matrix") 
plt.show()


# In[29]:


df.head()


# In[30]:


df.drop(labels = ["RowNumber","CustomerId", "Surname"], axis = 1, inplace=True)


# In[31]:


df = pd.get_dummies(df, drop_first = True)


# In[32]:


df.head()


# In[33]:


X = df.drop("Exited", axis = 1)
y = df["Exited"]


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[36]:


Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)


# In[37]:


lr_model = LogisticRegression()


# In[38]:


lr_model.fit(X_train, y_train)


# In[39]:


lr_predictions = lr_model.predict(X_test)


# In[40]:


print("Logistic Regression Model:")
print(confusion_matrix(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))
print("Accuracy: ", accuracy_score(y_test, lr_predictions))
print("r2_Score: ", r2_score(y_test, lr_predictions))
print("Precision_score: ", precision_score(y_test, lr_predictions))
print("Recall_score: ", recall_score(y_test, lr_predictions))
print("f1_score: ", f1_score(y_test, lr_predictions))


# In[41]:


rf_model = RandomForestClassifier(n_estimators=1000, n_jobs = -1, random_state=42)


# In[42]:


rf_model.fit(X_train, y_train)


# In[43]:


rf_predictions = rf_model.predict(X_test)


# In[44]:


print("Random Forest Model:")
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))
print("Accuracy: ", accuracy_score(y_test, rf_predictions))
print("r2_Score: ", r2_score(y_test, rf_predictions))
print("Precision_score: ", precision_score(y_test, rf_predictions))
print("Recall_score: ", recall_score(y_test, rf_predictions))
print("f1_score: ", f1_score(y_test, rf_predictions))


# In[45]:


gb_model = GradientBoostingClassifier(n_estimators=1000, learning_rate = 0.02, max_depth = 1, random_state=42)


# In[46]:


gb_model.fit(X_train, y_train)


# In[47]:


gb_predictions = gb_model.predict(X_test)


# In[48]:


print("Gradient Boosting Model:")
print(confusion_matrix(y_test, gb_predictions))
print(classification_report(y_test, gb_predictions))
print("Accuracy: ", accuracy_score(y_test, gb_predictions))
print("r2_Score: ", r2_score(y_test, gb_predictions))
print("Precision_score: ", precision_score(y_test, gb_predictions))
print("Recall_score: ", recall_score(y_test, gb_predictions))
print("f1_score: ", f1_score(y_test, gb_predictions))


# In[ ]:




