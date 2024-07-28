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
sns.set_theme(style="white")
sns.set_theme(style="whitegrid", color_codes=True)
import researchpy as rp


#Reading in data from Social Progress Index (2015)

SPIndex = pd.read_excel(
    '/Users/jborri/Documents/GitHub/Community-Indicators/Logistic Regression Analysis/2015 Social Progress Index Data.xlsx',
    index_col='Country',
    na_values=['NA']
      )
SPIndex.head()

#Reading in data from Human Development Index (United Nations- 2015)

UNHDIndex = pd.read_excel(
    '/Users/jborri/Documents/GitHub/Community-Indicators/Logistic Regression Analysis/HumanDevelopment.xlsx',
    index_col='Country',
    na_values=['NA']
)
UNHDIndex.head()

#Reading in data from World Happiness Report (2019)

WHRIndex = pd.read_excel(
    '/Users/jborri/Documents/GitHub/Community-Indicators/Logistic Regression Analysis/WHINDEX2019.xls',
    index_col='Country',
    na_values=['NA']
)
WHRIndex.head()

#Reading in data from Multidimensional Poverty Index (United Nations 2015)

MPIndex = pd.read_csv(
    '/Users/jborri/Documents/GitHub/Community-Indicators/Logistic Regression Analysis/multidimensional_poverty.csv',
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

#Checking Indicator dtypes to make sure they are all numeric (they aren't but)
print(IndexTotal.dtypes)

#Describing Merged DF for our dependent variable.
#WHI defines social support as follows:  "Social support (or having someone to count on in times of trouble) is the national
#average of the binary responses (either 0 or 1) to the GWP question “If you
#were in trouble, do you have relatives or friends you can count on to help you
#whenever you need them, or not?”"

IndexTotal['Social support'].describe()

#Creating Binary Variable for Social Support, (using 25th percentile)

IndexTotal['SS Binary'] = 0
IndexTotal.loc[IndexTotal['Social support'] < 0.739719,
               ['SS Binary']
               ] = 1 #Nations with lower SS value will be assigned 1
IndexTotal['SS Binary'].describe()

IndexTotal['SS Binary']

#Creating Logit Regression Model for Health

Y = IndexTotal['SS Binary']
X = IndexTotal[['Undernourishment',
                'Maternal mortality rate',
                'Child mortality rate',
                'Life expectancy',
                'Healthy life expectancy at birth',
                'Health Deprivation'
                
                ]]
model0 = sm.Logit(Y, X, missing='drop').fit()
print(model0.summary())

#exponentiating to interpret as odds.

print(math.exp(model0.params[0]), math.exp(model0.params[1]), math.exp(model0.params[2]), math.exp(model0.params[3]), 1/math.exp(model0.params[4]), 1/math.exp(model0.params[5]))

model0_marginals = model0.get_margeff()
print(model0_marginals.summary())

model0_marginals = model0.get_margeff(at='median')
print(model0_marginals.summary())

model0_pred = model0.pred_table()
print(model0_pred) # Correct predictions are on the diagonal of the 2d array.

correct_i = 31 / (31 + 6) # The proportion of correct predictions of 0.
correct_j = 19 / (19 + 7) # The proportion of correct predictions of 1.
print(correct_i, correct_j)

sns.countplot(x='SS Binary', data=IndexTotal)
plt.title('Sense of Community for Select Countries (Binary)')
plt.show()
plt.savefig('Count_plot')

p = sns.displot(IndexTotal, x='Undernourishment', hue='SS Binary', multiple="stack", height=6, aspect=1.25)
plt.title('Social Support by Undernourishment')
plt.show()

p = sns.displot(IndexTotal, x='Health Deprivation', hue='SS Binary', multiple="stack", height=6, aspect=1.25)
plt.title('Social Support by Undernourishment')
plt.show()

#Regression Model for Economic/Poverty

Y = IndexTotal['SS Binary']
X = IndexTotal[['Access to piped water',
                'Availability of affordable housing',
                'Depth of food deficit',
                'Human Development Index (HDI)',
                'Gross National Income (GNI) per Capita',
                'gini of household income reported in Gallup, by wp5-year',
                'Multidimensional Poverty Index (MPI, 2010)',
                'Living Standards',
                'Population Below National Poverty Line'
                
                ]]
model1 = sm.Logit(Y, X, missing='drop').fit()
print(model1.summary())

#exponentiating to interpret as odds.
#take inverse if negative

print(1/math.exp(model1.params[0]), 1/math.exp(model1.params[1]), math.exp(model1.params[2]), math.exp(model1.params[3]), 1/math.exp(model1.params[4]), math.exp(model1.params[5]),math.exp(model1.params[6]), 1/math.exp(model1.params[7]), 1/math.exp(model1.params[8]))

model1_marginals = model1.get_margeff()
print(model1_marginals.summary())

model1_marginals = model1.get_margeff(at='median')
print(model1_marginals.summary())

model1_pred = model1.pred_table()
print(model1_pred) 

correct_i = 28 / (28 + 5) # The proportion of correct predictions of 0.
correct_j = 21 / (21 + 4) # The proportion of correct predictions of 1.
print(correct_i, correct_j)

#Regression Model for Government

Y = IndexTotal['SS Binary']
X = IndexTotal[['Press Freedom Index',
                'Political rights',
                'Freedom of speech',
                'Private property rights',
                'Freedom over life choices',
                'Corruption',
                'Freedom to make life choices',
                'Perceptions of corruption',
                
                
                ]]
model2 = sm.Logit(Y, X, missing='drop').fit()
print(model2.summary())

#exponentiating to interpret as odds.
#take inverse if negative

print(math.exp(model2.params[0]), math.exp(model2.params[1]), math.exp(model2.params[2]), 1/math.exp(model2.params[3]), 1/math.exp(model2.params[4]), 1/math.exp(model2.params[5]), 1/math.exp(model2.params[6]), 1/math.exp(model2.params[7]))

model2_marginals = model2.get_margeff()
print(model2_marginals.summary())

model2_marginals = model2.get_margeff(at='median')
print(model2_marginals.summary())

model2_pred = model2.pred_table()
print(model2_pred) 

correct_i = 85 / (85 + 5) # The proportion of correct predictions of 0.
correct_j = 22 / (22 + 8) # The proportion of correct predictions of 1.
print(correct_i, correct_j)

#Regression Model for Happiness/Community Values

Y = IndexTotal['SS Binary']
X = IndexTotal[['Homicide rate',
                'Level of violent crime',
                'Perceived criminality',
                'Internet users',
                'Suicide rate',
                'Life Ladder',
                'Generosity',
                'Positive affect',
                
                
                ]]
model3 = sm.Logit(Y, X, missing='drop').fit()
print(model3.summary())

#exponentiating to interpret as odds.
#take inverse if negative

print(1/math.exp(model3.params[0]), math.exp(model3.params[1]), math.exp(model3.params[2]), 1/math.exp(model3.params[3]), 1/math.exp(model3.params[4]), 1/math.exp(model3.params[5]), math.exp(model3.params[6]), math.exp(model3.params[6]))

model3_marginals = model3.get_margeff()
print(model3_marginals.summary())

model3_marginals = model3.get_margeff(at='median')
print(model3_marginals.summary())

model3_pred = model3.pred_table()

print(model3_pred)
correct_i = 83 / (83 + 7) # The proportion of correct predictions of 0.
correct_j = 17 / (13 + 13) # The proportion of correct predictions of 1.
print(correct_i, correct_j)