# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn
import scipy
import matplotlib.pyplot as plt

pd.set_option('display.width', 100)

_data = pd.read_csv('nesarc_pds.csv', low_memory=False)
_data['S3AQ3B1'] = pd.to_numeric(_data['S3AQ3B1'], errors='coerce')
_data['S3AQ3C1'] = pd.to_numeric(_data['S3AQ3C1'], errors='coerce')
_data['CHECK321'] = pd.to_numeric(_data['CHECK321'], errors='coerce')
#AGE WHEN SMOKED FIRST FULL CIGARETTE
_data['S3AQ2A1'] = pd.to_numeric(_data['S3AQ2A1'], errors='coerce')

#subset data to young adults age 18 to 100 who have smoked in the past 12 months
_sub1 = _data.copy()[(_data['AGE']>=18) & (_data['AGE']<=100) & (_data['CHECK321']==1)]

#SETTING MISSING DATA
_sub1['S3AQ3B1'] = _sub1['S3AQ3B1'].replace(9, np.nan)
_sub1['S3AQ3C1'] = _sub1['S3AQ3C1'].replace(99, np.nan)
_sub1['S3AQ2A1'] = _sub1['S3AQ2A1'].replace(' ', np.nan)
_sub1['S3AQ2A1'] = _sub1['S3AQ2A1'].replace(99, np.nan)

#recoding number of days smoked in the past month
_recode1 = {1: 30, 2: 22, 3: 14, 4: 5, 5: 2.5, 6: 1}
_sub1.loc[:, 'USFREQMO']= _sub1['S3AQ3B1'].map(_recode1)
_sub1.loc[:, 'USFREQMO']= pd.to_numeric(_sub1['USFREQMO'], errors='coerce')

# Creating a secondary variable multiplying the days smoked/month and the number of cig/per day
_sub1.loc[:, 'NUMCIGMO_EST'] = _sub1['USFREQMO'] * _sub1['S3AQ3C1']
_sub1.loc[:, 'NUMCIGMO_EST'] = pd.to_numeric(_sub1['NUMCIGMO_EST'], errors='coerce')


_scat1 = seaborn.regplot(x="S3AQ2A1", y="NUMCIGMO_EST", fit_reg=True, data=_sub1)
plt.xlabel('Age when smoked first full cigarette')
plt.ylabel('Number of cig per month')
plt.title('Scatterplot for the Association Between first age and num of cig/month')
plt.show()

_data_clean=_sub1.dropna()

print('************')
print ('association between first age and num of cig/month')
_result1 = scipy.stats.pearsonr(_data_clean['S3AQ2A1'], _data_clean['NUMCIGMO_EST'])
print (_result1)
print ('R='+str(_result1[0]))
print ('R-square='+str((_result1[0])**2))
print ('p='+str(_result1[1]))

