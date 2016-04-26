#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

def sigmoid(_z):
	return 1.0/ (1.0 + np.exp(-_z))

def CostFunction_logistic(_X, _y, _weight, _bias):
	_M = len(_y)
	_hypo = sigmoid(_X.dot(_weight) + _bias)
	_J =  (- np.dot(_y.T, np.log(_hypo)) - np.dot(1-_y.T, np.log(1-_hypo))).sum()/_M
	return _J

def evaluate_theta_logistics(_X, _y, _predictY):
	_M = len(_y)
	_judge = np.zeros((_M,1))
	_judge[_y == _predictY] = 1
	_accuracy = float(_judge.sum())/float(_M)
	_precision = float((_y * _predictY).sum())/ float(_predictY.sum())
	_recall = float((_y * _predictY).sum())/ float(_y.sum())
	_F_score = 2.0 * _precision * _recall / (_precision + _recall)
	return _accuracy, _precision, _recall, _F_score


print '*************************'
print '***ex2data1***'
_data = pd.read_csv('ex2data1.txt', header=None)
_X = np.array([_data[0], _data[1]]).T
_y = np.array(_data[2])
_pos = (_y==1)
plt.scatter(_X[_pos,0], _X[_pos,1], marker='+', c='b')
_neg = (_y==0)
plt.scatter(_X[_neg,0], _X[_neg,1], marker='o', c='y')
plt.legend(['Admitted', 'Not addmitted'], scatterpoints=1)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')

print '***Start Fitting***'
_model = linear_model.LogisticRegression(C=1000000.0)
_model.fit(_X, _y)
print 'Coefficients:', _model.coef_
print 'Intercept:', _model.intercept_
print 'R2:', _model.score(_X, _y)
print 'J:', CostFunction_logistic(_X, _y, _model.coef_.reshape(-1,1), _model.intercept_[0])
_accuracy, _precision, _recall, _F_score = evaluate_theta_logistics(_X, _y, _model.predict(_X))
print 'accuracy=' +str(np.round(_accuracy,3))
print 'precision=' +str(np.round(_precision,3))
print 'recall=' +str(np.round(_recall,3))
print 'F_score=' +str(np.round(_F_score,3))

_plot_x = np.array([min(_X[:,0])-2, max(_X[:,0])+2])
_plot_y = - (_model.intercept_[0] + _model.coef_[0][0] * _plot_x)  / _model.coef_[0][1]
plt.plot(_plot_x, _plot_y, 'b')
plt.show ()
print 'for [45, 85], we predict ', _model.predict(np.array([[45.0,85.0]]))
