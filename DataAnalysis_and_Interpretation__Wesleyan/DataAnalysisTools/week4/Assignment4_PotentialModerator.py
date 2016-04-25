# -*- coding: utf-8 -*-
import pandas as pd  
import numpy as np
import scipy.stats
import seaborn
import matplotlib.pyplot as plt

pd.set_option('display.width', 100)

print('***********************************************')
print('***********************************************')
print('***********************************************')
#_data = pd.read_csv('nesarc_pds.csv', low_memory=False)

_data['MARITAL'] = pd.to_numeric(_data['MARITAL'], errors='coerce')
_data['AGE'] = pd.to_numeric(_data['AGE'], errors='coerce')
#Alcohol abuse/dependence happen in the last 12 month
_data['S2BQ1B1'] = pd.to_numeric(_data['S2BQ1B1'], errors='coerce')

#subset _data to adults age 20 to 50
_sub1=_data[(_data['AGE']>=20) & (_data['AGE']<=50)]

#make a copy of my new subsetted _data
_sub2 = _sub1.copy()

_recode1 = {1:'Married', 2:'Liv some', 3:'Widow', 4:'Divorced', 5:'Separa', 6:'Nev Marr'}
_sub2.loc[:, 'MARITAL']= _sub2['MARITAL'].map(_recode1)
_label = ['Married', 'Liv some', 'Widow', 'Divorced', 'Separa', 'Nev Marr']

_sub2.loc[:, 'SEX']= _sub2['SEX'].map({1:'Male', 2:'Female'})

_recode2 = {1:'Yes', 2: 'No'}
_sub2.loc[:, 'ABUSE']= _sub2['S2BQ1B1'].map(_recode2)

# contingency table of observed counts
print('***Both Gender********************************************')
print('Contingency Table')
_ct1=pd.crosstab(_sub2['ABUSE'], _sub2['MARITAL'])
print (_ct1)
print('')

# column percentages
print('column percentages')
print( _ct1/_ct1.sum(axis=0) )
print('')

# chi-square
print ('chi-square value, p value, expected counts')
_cs1= scipy.stats.chi2_contingency(_ct1)
print (_cs1)
print('x2='+str(_cs1[0]))
print('p='+str(_cs1[1]))
print('')

_sub3 = _sub2.copy()
_sub3.loc[:, 'ABUSE']= _sub3['ABUSE'].map({'Yes':1, 'No':0})
_sub3["MARITAL"] = _sub3["MARITAL"].astype('category')
_sub3['ABUSE'] = pd.to_numeric(_sub3['ABUSE'], errors='coerce')
seaborn.factorplot(x="MARITAL", y="ABUSE", data=_sub3, kind="bar", ci=None)
plt.xlabel('MARITAL')
plt.ylabel('Alcohol abuse/dependence happen in last 12 month')
plt.title('Factor Plot: Both gender')
plt.show()
seaborn.factorplot(x="MARITAL", y="ABUSE", data=_sub3, kind="bar", ci=None, hue='SEX')
plt.xlabel('MARITAL')
plt.ylabel('Alcohol abuse/dependence happen in last 12 month')
plt.title('Factor Plot: Each gender')
plt.show()

#_recode1 = {1:'Married', 2:'Liv some', 3:'Widow', 4:'Divorced', 5:'Separa', 6:'Nev Marr'}
_p = np.ndarray((6,6), dtype=np.float)
_x2 = np.ndarray((6,6), dtype=np.float)
for _i in range(1,7):
	for _j in range(1,_i):
		_sub4 = _sub1.copy()
		_recode3 = {_i: _recode1[_i], _j: _recode1[_j]}
		_sub4.loc[:, 'MARITAL']= _sub4['MARITAL'].map(_recode3)
		_sub4.loc[:, 'ABUSE']= _sub4['S2BQ1B1'].map(_recode2)
		_ct2=pd.crosstab(_sub4['ABUSE'], _sub4['MARITAL'])
		_cs2= scipy.stats.chi2_contingency(_ct2)
		_p[_i-1,_j-1] = _cs2[1]
		_p[_j-1,_i-1] = _cs2[1]
		_x2[_i-1,_j-1] = _cs2[0]
		_x2[_j-1,_i-1] = _cs2[0]
for _i in range(1,7):
	_p[_i-1,_i-1] = 1.0
	_x2[_i-1,_i-1] = np.nan

print('Chi-Square')
_x2_pd = pd.DataFrame(_x2, columns=_label, index=_label)
print(_x2_pd)

print('')
print('P')
_p_pd = pd.DataFrame(_p, columns=_label, index=_label)
print(_p_pd)

_m = 6*5/2
_BonferroniAdjustment = 0.05 / _m
print('')
print('Number of Comparison='+str(_m))
print('BonferroniAdjustment='+str(_BonferroniAdjustment))
_tf = (_p<_BonferroniAdjustment)
_tf_pd = pd.DataFrame(_tf, columns=_label, index=_label)
print('')
print('Statistically significant')
print(_tf_pd)

print('***Male********************************************')
_sub5 = _sub2.copy()
_sub5 = _sub5[_sub5['SEX']=='Male']
print('Contingency Table')
_ct5=pd.crosstab(_sub5['ABUSE'], _sub5['MARITAL'])
print (_ct5)
print('')
print('column percentages')
print( _ct5/_ct5.sum(axis=0) )
print('')
print ('chi-square value, p value, expected counts')
_cs5= scipy.stats.chi2_contingency(_ct5)
print (_cs5)
print('x2='+str(_cs5[0]))
print('p='+str(_cs5[1]))
print('')
#_recode1 = {1:'Married', 2:'Liv some', 3:'Widow', 4:'Divorced', 5:'Separa', 6:'Nev Marr'}
_p = np.ndarray((6,6), dtype=np.float)
_x2 = np.ndarray((6,6), dtype=np.float)
for _i in range(1,7):
	for _j in range(1,_i):
		_sub4 = _sub1.copy()
		_sub4 = _sub4[_sub4['SEX']==1]
		_recode3 = {_i: _recode1[_i], _j: _recode1[_j]}
		_sub4.loc[:, 'MARITAL']= _sub4['MARITAL'].map(_recode3)
		_sub4.loc[:, 'ABUSE']= _sub4['S2BQ1B1'].map(_recode2)
		_ct2=pd.crosstab(_sub4['ABUSE'], _sub4['MARITAL'])
		_cs2= scipy.stats.chi2_contingency(_ct2)
		_p[_i-1,_j-1] = _cs2[1]
		_p[_j-1,_i-1] = _cs2[1]
		_x2[_i-1,_j-1] = _cs2[0]
		_x2[_j-1,_i-1] = _cs2[0]
for _i in range(1,7):
	_p[_i-1,_i-1] = 1.0
	_x2[_i-1,_i-1] = np.nan

print('Chi-Square')
_x2_pd = pd.DataFrame(_x2, columns=_label, index=_label)
print(_x2_pd)

print('')
print('P')
_p_pd = pd.DataFrame(_p, columns=_label, index=_label)
print(_p_pd)

_m = 6*5/2
_BonferroniAdjustment = 0.05 / _m
print('')
print('Number of Comparison='+str(_m))
print('BonferroniAdjustment='+str(_BonferroniAdjustment))
_tf = (_p<_BonferroniAdjustment)
_tf_pd = pd.DataFrame(_tf, columns=_label, index=_label)
print('')
print('Statistically significant')
print(_tf_pd)

print('***Female********************************************')
_sub5 = _sub2.copy()
_sub5 = _sub5[_sub5['SEX']=='Female']
print('Contingency Table')
_ct5=pd.crosstab(_sub5['ABUSE'], _sub5['MARITAL'])
print (_ct5)
print('')
print('column percentages')
print( _ct5/_ct5.sum(axis=0) )
print('')
print ('chi-square value, p value, expected counts')
_cs5= scipy.stats.chi2_contingency(_ct5)
print (_cs5)
print('x2='+str(_cs5[0]))
print('p='+str(_cs5[1]))
print('')

#_recode1 = {1:'Married', 2:'Liv some', 3:'Widow', 4:'Divorced', 5:'Separa', 6:'Nev Marr'}
_p = np.ndarray((6,6), dtype=np.float)
_x2 = np.ndarray((6,6), dtype=np.float)
for _i in range(1,7):
	for _j in range(1,_i):
		_sub4 = _sub1.copy()
		_sub4 = _sub4[_sub4['SEX']==2]
		_recode3 = {_i: _recode1[_i], _j: _recode1[_j]}
		_sub4.loc[:, 'MARITAL']= _sub4['MARITAL'].map(_recode3)
		_sub4.loc[:, 'ABUSE']= _sub4['S2BQ1B1'].map(_recode2)
		_ct2=pd.crosstab(_sub4['ABUSE'], _sub4['MARITAL'])
		_cs2= scipy.stats.chi2_contingency(_ct2)
		_p[_i-1,_j-1] = _cs2[1]
		_p[_j-1,_i-1] = _cs2[1]
		_x2[_i-1,_j-1] = _cs2[0]
		_x2[_j-1,_i-1] = _cs2[0]
for _i in range(1,7):
	_p[_i-1,_i-1] = 1.0
	_x2[_i-1,_i-1] = np.nan

print('Chi-Square')
_x2_pd = pd.DataFrame(_x2, columns=_label, index=_label)
print(_x2_pd)

print('')
print('P')
_p_pd = pd.DataFrame(_p, columns=_label, index=_label)
print(_p_pd)

_m = 6*5/2
_BonferroniAdjustment = 0.05 / _m
print('')
print('Number of Comparison='+str(_m))
print('BonferroniAdjustment='+str(_BonferroniAdjustment))
_tf = (_p<_BonferroniAdjustment)
_tf_pd = pd.DataFrame(_tf, columns=_label, index=_label)
print('')
print('Statistically significant')
print(_tf_pd)
