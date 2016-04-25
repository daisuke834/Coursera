#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

print '*************************'
print '***ex1data1***'
_data = pd.read_csv('ex1data1.txt', header=None)
plt.scatter(_data[0], _data[1], marker='x', c='r')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
_X = np.array([_data[0]]).T
_y = np.array(_data[1])
_model = linear_model.LinearRegression()
_model.fit(_X, _y)
print 'Coefficients:', _model.coef_
print 'Intercept:', _model.intercept_
print 'R2:', _model.score(_X, _y)
print 'J:', np.mean( (_model.predict(_X) - _y)**2 ) / 2.0
_px = np.arange(_X.min(), _X.max(), 0.01)[:,np.newaxis]
_py = _model.predict(_px)
plt.plot(_px, _py, color='blue', linewidth=3)
plt.show()
print 'Predict: For population = 35,000, we predict a profit of', _model.predict(3.5)
print 'Predict: For population = 70,000, we predict a profit of', _model.predict(7.0)


print '*************************'
print '***ex1data2***'
_data = pd.read_csv('ex1data2.txt', header=None)
_X = np.array([_data[0],_data[1]]).T
_y = np.array(_data[2])
_model = linear_model.LinearRegression()
_model.fit(_X, _y)
print '\tfor [1650, 3], we predict $', _model.predict(np.array([[1650,3]]))
print 'Coefficients:', _model.coef_
print 'Intercept:', _model.intercept_
print 'R2:', _model.score(_X, _y)
print 'J:', np.mean( (_model.predict(_X) - _y)**2 ) / 2.0

