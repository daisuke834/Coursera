# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

if _data is None:
    _data = pd.read_csv('nesarc_pds.csv', low_memory=False)

#setting variables you will be working with to numeric
_data['S3AQ3B1'] = pd.to_numeric(_data['S3AQ3B1'], errors='coerce')
_data['S3AQ3C1'] = pd.to_numeric(_data['S3AQ3C1'], errors='coerce')
_data['CHECK321'] = pd.to_numeric(_data['CHECK321'], errors='coerce')

#subset data to young adults age 18 to 25 who have smoked in the past 12 months
_sub1 = _data[(_data['AGE']>=18) & (_data['AGE']<=25) & (_data['CHECK321']==1)]

#recoding number of days smoked in the past month
_recode1 = {1: 30, 2: 22, 3: 14, 4: 5, 5: 2.5, 6: 1}
_sub1.loc[:, 'USFREQMO']= _sub1['S3AQ3B1'].map(_recode1)

#converting new variable USFREQMMO to numeric
_sub1.loc[:, 'USFREQMO']= pd.to_numeric(_sub1['USFREQMO'], errors='coerce')

#Encoding Explanatory Variable to Text
_recode2 = {1:'Married', 2:'Liv some', 3:'Widow', 4:'Divorced', 5:'Separa', 6:'Nev Marr'}
_sub1.loc[:, 'MARITAL']= _sub1['MARITAL'].map(_recode2)

# Creating a secondary variable multiplying the days smoked/month and the number of cig/per day
_sub1.loc[:, 'NUMCIGMO_EST'] = _sub1['USFREQMO'] * _sub1['S3AQ3C1']
_sub1.loc[:, 'NUMCIGMO_EST'] = pd.to_numeric(_sub1['NUMCIGMO_EST'], errors='coerce')

# BoxPlot
_sub1.boxplot(column='NUMCIGMO_EST', by='MARITAL')

# using ols function for calculating the F-statistic and associated p value
_model1 = smf.ols(formula='NUMCIGMO_EST ~ C(MARITAL)', data=_sub1)
_results1 = _model1.fit()
print (_results1.summary())

_sub2 = _sub1[['NUMCIGMO_EST', 'MARITAL']].dropna()

print ('means for numcigmo_est by major depression status')
_m1= _sub2.groupby('MARITAL').mean()
print (_m1)

print ('standard deviations for numcigmo_est by major depression status')
_sd1 = _sub2.groupby('MARITAL').std()
print (_sd1)

_mc1 = multi.MultiComparison(_sub2['NUMCIGMO_EST'], _sub2['MARITAL'])
_res1 = _mc1.tukeyhsd()
print(_res1.summary())

