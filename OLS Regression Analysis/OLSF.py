#Importing Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import math
import researchpy as rp
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import scipy.stats as scp

#Reading in data from Social Progress Index (2015)

SPIndex = pd.read_excel(
    '/Users/jborri/Documents/GitHub/Community-Indicators/OLS Regression Analysis/2015 Social Progress Index Data.xlsx',
    index_col='Country',
    na_values=['NA']
      )
SPIndex.head()

#Reading in data from Human Development Index (United Nations- 2015)

UNHDIndex = pd.read_excel(
    '//Users/jborri/Documents/GitHub/Community-Indicators/OLS Regression Analysis/HumanDevelopment.xlsx',
    index_col='Country',
    na_values=['NA']
)
UNHDIndex.head()

#Reading in data from World Happiness Report (2019)

WHRIndex = pd.read_excel(
    '/Users/jborri/Documents/GitHub/Community-Indicators/OLS Regression Analysis/WHINDEX2019.xls',
    index_col='Country',
    na_values=['NA']
)
WHRIndex.head()

#Reading in data from Multidimensional Poverty Index (United Nations 2015)

MPIndex = pd.read_csv(
    '/Users/jborri/Documents/GitHub/Community-Indicators/OLS Regression Analysis/multidimensional_poverty.csv',
    index_col='Country',
    na_values=['NA']
)
MPIndex.head()

#Merging DataFrames

IndexTotal = pd.concat([
    SPIndex,
    UNHDIndex,
    WHRIndex,
    MPIndex
    ],
    axis=1
    )
IndexTotal.shape

IndexTotal.head()

#Creating Linear Model of Merged Data

Y = IndexTotal['Social support'] # A measure of Quality of Community
X = IndexTotal[['Human Development Index (HDI)',
                'Life Ladder',
                'Community safety net',
                'Tolerance for immigrants',
                #'MPI HDRO Percent'


]]
X = sm.add_constant(X)
model0 = sm.OLS(Y, X, missing='drop').fit()
print(model0.summary())

X.corr()

# Checking on colinearity
corrtab, corrsig = scp.stats.pearsonr(IndexTotal['Social support'].dropna(), IndexTotal['Life Ladder'].dropna())
corrtab

corrsig

q = sns.lmplot(data=IndexTotal, x='Life Ladder', y='Social support')
q.figure.set_figwidth(10)
q.figure.set_figheight(6)
plt.title('Relationship Between Social Support and Life Ladder (Happiness Index)')
plt.show()

q = sns.lmplot(data=IndexTotal, x='Community safety net', y='Social support')
q.figure.set_figwidth(10)
q.figure.set_figheight(6)
plt.title('Relationship Between Social Support and Community Safety Net')
plt.show()

sns.pairplot(IndexTotal[['Human Development Index (HDI)',
                'Life Ladder',
                'Community safety net',
                'Tolerance for immigrants']])
plt.suptitle('Scatter Plot Matrix for Key Variables', y=1.02)
plt.show()

