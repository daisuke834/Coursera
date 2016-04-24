#syntax used to run an ANOVA
'''
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
'''
#output
'''
                          OLS Regression Results                            
==============================================================================
Dep. Variable:           NUMCIGMO_EST   R-squared:                       0.013
Model:                            OLS   Adj. R-squared:                  0.010
Method:                 Least Squares   F-statistic:                     5.457
Date:                Sat, 23 Apr 2016   Prob (F-statistic):           0.000230
Time:                        00:09:23   Log-Likelihood:                -12188.
No. Observations:                1703   AIC:                         2.439e+04
Df Residuals:                    1698   BIC:                         2.441e+04
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==========================================================================================
                           coef    std err          t      P>|t|      [95.0% Conf. Int.]
Intercept                397.7297     51.099      7.783      0.000       297.505   497.954
C(MARITAL)[T.Liv some]    -2.7918     57.588     -0.048      0.961      -115.742   110.159
C(MARITAL)[T.Married]    -38.0697     54.429     -0.699      0.484      -144.824    68.684
C(MARITAL)[T.Nev Marr]   -89.1731     51.875     -1.719      0.086      -190.918    12.572
C(MARITAL)[T.Separa]      45.4975     69.332      0.656      0.512       -90.487   181.482
==============================================================================
Omnibus:                     1180.678   Durbin-Watson:                   2.021
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            27339.877
Skew:                           2.931   Prob(JB):                         0.00
Kurtosis:                      21.733   Cond. No.                         19.7
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

means for numcigmo_est by major depression status
        NUMCIGMO_EST
MARITAL               
Divorced    397.729730
Liv some    394.937956
Married     359.660000
Nev Marr    308.556612
Separa      443.227273

standard deviations for numcigmo_est by major depression status
        NUMCIGMO_EST
MARITAL               
Divorced    283.380452
Liv some    403.778716
Married     314.815068
Nev Marr    290.855882
Separa      471.273182

Multiple Comparison of Means - Tukey HSD,FWER=0.05 
====================================================
group1   group2  meandiff   lower    upper   reject
Divorced Liv some -2.7918  -160.0466 154.463  False 
Divorced Married  -38.0697 -186.6977 110.5582 False 
Divorced Nev Marr -89.1731 -230.8275 52.4813  False 
Divorced  Separa  45.4975  -143.8265 234.8216 False 
Liv some Married  -35.278   -124.037  53.481  False 
Liv some Nev Marr -86.3813 -162.8919 -9.8708   True 
Liv some  Separa  48.2893   -98.7871 195.3658 False 
Married  Nev Marr -51.1034 -107.8049  5.5981  False 
Married   Separa  83.5673   -54.2467 221.3812 False 
Nev Marr  Separa  134.6707   4.4079  264.9334  True 
'''

# interpretation
ANOVA revealed that CURRENT MARITAL STATUS has a significant affect on number of cigarettes per month (F=5.457 and P=0.000230).
Post hoc comparisons of fmean number of current number of cigarettes per month revealed that those who have never been married tend to smoke (mean=308.6) significantly lower number of cigarettes per month than those who are living with someone as if married (mean=394.9) and those who are separated (mean=443.2).
![BoxPlot](boxplot.png)
