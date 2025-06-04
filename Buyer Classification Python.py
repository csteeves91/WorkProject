
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from kmodes.kprototypes import KPrototypes
from kneed import KneeLocator

#Step-2: Read and Load the Dataset

df = pd.read_csv('C:/Users/cjsteeves/Desktop/Buyer Classification/Updated Repeated Buyers.csv')
df.head()

y=sns.histplot(x=df['Max_Spent %'], bins=10, kde=True)
y.set(xlabel = 'Highest Percent of Product Line')
sns.boxplot(x=df['Max_Spent %'], showmeans=True)

sns.boxplot(x=df['Number of Orders'], y=df['Is_Seller?'], showmeans=True)
sns.displot(x='Number of Orders', row='Is_Seller?', data=df, linewidth=0, kde=True)

numerical = df[['Number of Orders','Total Spent','Total Qty','# of Product Lines','Average Cart Amount',
             'Average Item Value','Duration (month)','Monthly Spend','Monthly Qty','Monthly Checkouts',
             'Magic%','YGO%','PKM%',
             'Digimon%','FleshBlood%','OnePiece%','Comic%','Max_Spent %']]

categorical = df[['BUYER_ID','Is_Seller?','PRODUCT_LINE','Kind']]

summary = df[['Number of Orders','Total Spent','Total Qty','# of Product Lines','Average Cart Amount',
             'Average Item Value','Duration (month)','Monthly Spend','Monthly Qty','Monthly Checkouts','Magic%','YGO%','PKM%',
             'Digimon%','FleshBlood%','OnePiece%','Comic%','Max_Spent %']].describe()


summarysub = df.groupby('Is_Seller?')['Number of Orders','Total Spent','Total Qty','# of Product Lines','Average Cart Amount',
             'Average Item Value','Duration (month)','Monthly Spend','Monthly Qty','Monthly Checkouts'].describe()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from kmodes.kprototypes import KPrototypes
from kneed import KneeLocator


# Multi Repeated Buyer Clusters
df = pd.read_csv('C:/Users/cjsteeves/Desktop/Buyer Classification/Updated Repeated Buyer Multi.csv')
df.head()
df.info()

len(df['BUYER_ID'].unique())
df.drop(['BUYER_ID'], axis=1, inplace=True)
df.info()

# Inspect the categorical variables
df.select_dtypes('object').nunique()
df.describe()
df.isna().sum()

numerical = df[['# of Product Lines','Average Cart Amount',
             'Average Item Value','Duration (month)','Monthly Spend','Monthly Qty','Monthly Checkouts',
             'Max_Spent %']]

categorical = df[['Is_Seller?','PRODUCT_LINE']]

# check if any abonormal values in the categorical varaibles
for i in categorical.columns:
    print(categorical[i].unique())
    
# trasform categorical labels to numerical labels
encoders={}
for col_name in categorical.columns:
    series = categorical[col_name]
    label_encoder = LabelEncoder()
    categorical[col_name]=pd.Series(label_encoder.fit_transform(series[series.notnull()]),
    )                                
    encoders[col_name] = label_encoder
#handling missing values using knn for both categorical and numerical labels
#numerical values
imputer = KNNImputer(n_neighbors=5)
numerical.loc[:]=imputer.fit_transform(numerical)
#categorical values
imputer = KNNImputer(n_neighbors=1)
categorical.loc[:]=imputer.fit_transform(categorical)
#concatenate the data
data=pd.concat([categorical, numerical],axis=1)
#load data
pca=PCA(2)
#transform the data
d_f=pca.fit_transform(data)
d_f.shape
#decode categorical data because k-prototype works with raw categorical data
for i in categorical.columns:
    data[i]=data[i].astype(int)
for col_name in categorical:
    data[col_name]=encoders[col_name].inverse_transform(data[col_name])
data.info()
#data ready to be implemented using k-prototype algorithm 
#get the position of categorical columns
catcolpos=[data.columns.get_loc(col) for col in list(data.select_dtypes('object').columns)]
print ('Categorical columns          :{}'.format(list(data.select_dtypes('object').columns)))
print ('Categorical columns position :{}'.format(catcolpos))
#convert data frame to matrix
dfmatrix=data.to_numpy()

#elbow methods to detect number of k

cost=[]
for cluster in range (1,10):
    try:
        kprototype=KPrototypes(n_jobs=-1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(dfmatrix, categorical = catcolpos)
        cost.append(kprototype.cost_)
        print ('Cluster initiation: {}'.format(cluster))
    except:
        break
 
plt.plot(cost)    
plt.xlabel('K')
plt.ylabel('cost')
plt.show   


#cost(sum distance): confirm visual clue of elbow plot
#kneelocator class will detect elbows if curve is convex; if concave, will detect knees
cost_knee_c3 = KneeLocator(
    x=range (1,10),
    y=cost,
    s=0.1, curve ="convex", direction="decreasing", online=True)

k_cost_c3 = cost_knee_c3.elbow
print("elbow at k=", f'{k_cost_c3:.0f} clusters')

# build the kprototype model with 8 clusters
kprototype = KPrototypes(n_jobs=-1, n_clusters = 8, init = 'Huang', random_state = 0)
data['clusters'] = kprototype.fit_predict(dfmatrix, categorical = catcolpos)

#predict the labels of clusters
label = kprototype.fit_predict(dfmatrix, categorical = catcolpos)

#getting unique labels
u_labels = np.unique(label)
#plotting the results
for i in u_labels:
    plt.scatter(d_f[label == i, 0], d_f[label == i, 1], label = i)
plt.legend()
plt.show()

#the volume of each cluster
data['clusters'].value_counts().plot(kind='bar')
#stats of numerical data by mean
clusterdesc=data.groupby(['clusters']).mean()
 #stats of categorical data by mode
eachcluster = data.groupby(['clusters']).agg(lambda x:pd.Series.mode(x).iat[0])[['Is_Seller?','PRODUCT_LINE']]


# Multi New Buyer Clusters

df = pd.read_csv('C:/Users/cjsteeves/Desktop/Buyer Classification/New Buyer Multi.csv')
df.head()
df.info()

len(df['BUYER_ID'].unique())
df.drop(['BUYER_ID'], axis=1, inplace=True)
df.info()

# Inspect the categorical variables
df.select_dtypes('object').nunique()
df.describe()
df.isna().sum()

numerical = df[['# of Product Lines','Qty','Spend']]

categorical = df[['Is_Seller?']]

# check if any abonormal values in the categorical varaibles
for i in categorical.columns:
    print(categorical[i].unique())
    
# trasform categorical labels to numerical labels
encoders={}
for col_name in categorical.columns:
    series = categorical[col_name]
    label_encoder = LabelEncoder()
    categorical[col_name]=pd.Series(label_encoder.fit_transform(series[series.notnull()]),
    )                                
    encoders[col_name] = label_encoder
#handling missing values using knn for both categorical and numerical labels
#numerical values
imputer = KNNImputer(n_neighbors=5)
numerical.loc[:]=imputer.fit_transform(numerical)
#categorical values
imputer = KNNImputer(n_neighbors=1)
categorical.loc[:]=imputer.fit_transform(categorical)
#concatenate the data
data=pd.concat([categorical, numerical],axis=1)
#load data
pca=PCA(2)
#transform the data
d_f=pca.fit_transform(data)
d_f.shape
#decode categorical data because k-prototype works with raw categorical data
for i in categorical.columns:
    data[i]=data[i].astype(int)
for col_name in categorical:
    data[col_name]=encoders[col_name].inverse_transform(data[col_name])
data.info()
#data ready to be implemented using k-prototype algorithm 
#get the position of categorical columns
catcolpos=[data.columns.get_loc(col) for col in list(data.select_dtypes('object').columns)]
print ('Categorical columns          :{}'.format(list(data.select_dtypes('object').columns)))
print ('Categorical columns position :{}'.format(catcolpos))
#convert data frame to matrix
dfmatrix=data.to_numpy()

#elbow methods to detect number of k

cost=[]
for cluster in range (1,10):
    try:
        kprototype=KPrototypes(n_jobs=-1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(dfmatrix, categorical = catcolpos)
        cost.append(kprototype.cost_)
        print ('Cluster initiation: {}'.format(cluster))
    except:
        break
 
plt.plot(cost)    
plt.xlabel('K')
plt.ylabel('cost')
plt.show   


# build the kprototype model with 4 clusters
kprototype = KPrototypes(n_jobs=-1, n_clusters = 4, init = 'Huang', random_state = 0)
data['clusters'] = kprototype.fit_predict(dfmatrix, categorical = catcolpos)

#predict the labels of clusters
label = kprototype.fit_predict(dfmatrix, categorical = catcolpos)

#getting unique labels
u_labels = np.unique(label)
#plotting the results
for i in u_labels:
    plt.scatter(d_f[label == i, 0], d_f[label == i, 1], label = i)
plt.legend()
plt.show()

#the volume of each cluster
data['clusters'].value_counts().plot(kind='bar')
#stats of numerical data by mean
clusterdesc=data.groupby(['clusters']).mean()
 #stats of categorical data by mode
eachcluster = data.groupby(['clusters']).agg(lambda x:pd.Series.mode(x).iat[0])[['Is_Seller?']]


# Single New Buyer Clusters

df = pd.read_csv('C:/Users/cjsteeves/Desktop/Buyer Classification/New Buyer Single.csv')
df.head()
df.info()

len(df['BUYER_ID'].unique())
df.drop(['BUYER_ID'], axis=1, inplace=True)
df.drop(['# of Product Lines'], axis=1, inplace=True)
df.info()

# Inspect the categorical variables
df.select_dtypes('object').nunique()
df.describe()
df.isna().sum()

numerical = df[['Qty','Spend']]

categorical = df[['Is_Seller?', 'PRODUCT_LINE']]

# check if any abonormal values in the categorical varaibles
for i in categorical.columns:
    print(categorical[i].unique())
    
# trasform categorical labels to numerical labels
encoders={}
for col_name in categorical.columns:
    series = categorical[col_name]
    label_encoder = LabelEncoder()
    categorical[col_name]=pd.Series(label_encoder.fit_transform(series[series.notnull()]),
    )                                
    encoders[col_name] = label_encoder
#handling missing values using knn for both categorical and numerical labels
#numerical values
imputer = KNNImputer(n_neighbors=5)
numerical.loc[:]=imputer.fit_transform(numerical)
#categorical values
imputer = KNNImputer(n_neighbors=1)
categorical.loc[:]=imputer.fit_transform(categorical)
#concatenate the data
data=pd.concat([categorical, numerical],axis=1)
#load data
pca=PCA(2)
#transform the data
d_f=pca.fit_transform(data)
d_f.shape
#decode categorical data because k-prototype works with raw categorical data
for i in categorical.columns:
    data[i]=data[i].astype(int)
for col_name in categorical:
    data[col_name]=encoders[col_name].inverse_transform(data[col_name])
data.info()
#data ready to be implemented using k-prototype algorithm 
#get the position of categorical columns
catcolpos=[data.columns.get_loc(col) for col in list(data.select_dtypes('object').columns)]
print ('Categorical columns          :{}'.format(list(data.select_dtypes('object').columns)))
print ('Categorical columns position :{}'.format(catcolpos))
#convert data frame to matrix
dfmatrix=data.to_numpy()

#elbow methods to detect number of k

cost=[]
for cluster in range (1,10):
    try:
        kprototype=KPrototypes(n_jobs=-1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(dfmatrix, categorical = catcolpos)
        cost.append(kprototype.cost_)
        print ('Cluster initiation: {}'.format(cluster))
    except:
        break
 
plt.plot(cost)    
plt.xlabel('K')
plt.ylabel('cost')
plt.show   


# build the kprototype model with 6 clusters
kprototype = KPrototypes(n_jobs=-1, n_clusters = 6, init = 'Huang', random_state = 0)
data['clusters'] = kprototype.fit_predict(dfmatrix, categorical = catcolpos)

#predict the labels of clusters
label = kprototype.fit_predict(dfmatrix, categorical = catcolpos)

#getting unique labels
u_labels = np.unique(label)
#plotting the results
for i in u_labels:
    plt.scatter(d_f[label == i, 0], d_f[label == i, 1], label = i)
plt.legend()
plt.show()

#the volume of each cluster
data['clusters'].value_counts().plot(kind='bar')
#stats of numerical data by mean
clusterdesc=data.groupby(['clusters']).mean()
 #stats of categorical data by mode
eachcluster = data.groupby(['clusters']).agg(lambda x:pd.Series.mode(x).iat[0])[['Is_Seller?']]




# Single Repeated Buyer Clusters

df = pd.read_csv('C:/Users/cjsteeves/Desktop/Buyer Classification/Updated Repeated Buyer.csv')
df.head()
df.info()

len(df['BUYER_ID'].unique())
df.drop(['BUYER_ID'], axis=1, inplace=True)
df.drop(['Max_Spent %'], axis=1, inplace=True)
df.info()

# Inspect the categorical variables
df.select_dtypes('object').nunique()
df.describe()
df.isna().sum()

numerical = df[['Average Cart Amount',
             'Average Item Value','Duration (month)','Monthly Spend','Monthly Qty','Monthly Checkouts']]

categorical = df[['Is_Seller?', 'PRODUCT_LINE']]

# check if any abonormal values in the categorical varaibles
for i in categorical.columns:
    print(categorical[i].unique())
    
# trasform categorical labels to numerical labels
encoders={}
for col_name in categorical.columns:
    series = categorical[col_name]
    label_encoder = LabelEncoder()
    categorical[col_name]=pd.Series(label_encoder.fit_transform(series[series.notnull()]),
    )                                
    encoders[col_name] = label_encoder
#handling missing values using knn for both categorical and numerical labels
#numerical values
imputer = KNNImputer(n_neighbors=5)
numerical.loc[:]=imputer.fit_transform(numerical)
#categorical values
imputer = KNNImputer(n_neighbors=1)
categorical.loc[:]=imputer.fit_transform(categorical)
#concatenate the data
data=pd.concat([categorical, numerical],axis=1)
#load data
pca=PCA(2)
#transform the data
d_f=pca.fit_transform(data)
d_f.shape
#decode categorical data because k-prototype works with raw categorical data
for i in categorical.columns:
    data[i]=data[i].astype(int)
for col_name in categorical:
    data[col_name]=encoders[col_name].inverse_transform(data[col_name])
data.info()
#data ready to be implemented using k-prototype algorithm 
#get the position of categorical columns
catcolpos=[data.columns.get_loc(col) for col in list(data.select_dtypes('object').columns)]
print ('Categorical columns          :{}'.format(list(data.select_dtypes('object').columns)))
print ('Categorical columns position :{}'.format(catcolpos))
#convert data frame to matrix
dfmatrix=data.to_numpy()

#elbow methods to detect number of k

cost=[]
for cluster in range (1,10):
    try:
        kprototype=KPrototypes(n_jobs=-1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(dfmatrix, categorical = catcolpos)
        cost.append(kprototype.cost_)
        print ('Cluster initiation: {}'.format(cluster))
    except:
        break
 
plt.plot(cost)    
plt.xlabel('K')
plt.ylabel('cost')
plt.show   


# build the kprototype model with 6 clusters
kprototype = KPrototypes(n_jobs=-1, n_clusters = 6, init = 'Huang', random_state = 0)
data['clusters'] = kprototype.fit_predict(dfmatrix, categorical = catcolpos)

#predict the labels of clusters
label = kprototype.fit_predict(dfmatrix, categorical = catcolpos)

#getting unique labels
u_labels = np.unique(label)
#plotting the results
for i in u_labels:
    plt.scatter(d_f[label == i, 0], d_f[label == i, 1], label = i)
plt.legend()
plt.show()

#the volume of each cluster
data['clusters'].value_counts().plot(kind='bar')
#stats of numerical data by mean
clusterdesc=data.groupby(['clusters']).mean()
 #stats of categorical data by mode
eachcluster = data.groupby(['clusters']).agg(lambda x:pd.Series.mode(x).iat[0])[['Is_Seller?']]

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
data = pd.read_csv('C:/Users/cjsteeves/Desktop/Buyer Classification/Updated Repeated Buyer.csv')
data = data[data['PRODUCT_LINE']=='Magic']
Spend = data['Monthly Spend'] 
Orders = data['Monthly Checkouts']
sns.boxplot(Spend)
sns.distplot(Spend)
sns.distplot(Orders)









