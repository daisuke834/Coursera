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

def Xshaping(_X):
	_X_mean = np.mean(_X, axis=0).reshape(1, -1)
	_X_std = np.std(_X, axis=0).reshape(1, -1)
	_Xnorm = ( _X - _X_mean ) / _X_std
	return _Xnorm, _X_mean, _X_std

def mapFeature(_X1, _X2, _degree):
	_degree = 6
	_X = None
	_X1 = _X1.reshape(len(_X1),1)
	_X2 = _X2.reshape(len(_X2),1)
	_i=1
	_j=0
	while _i<=_degree:
		_j=0
		while _j<=_i:
			if _X is None:
				_X = _X1**(_i-_j) * _X2**_j
			else:
				_X = np.hstack((_X, _X1**(_i-_j) * _X2**_j))
			_j=_j+1
		_i=_i+1
	return _X


print '*************************'
print '***ex2data2***'
_data = pd.read_csv('ex2data2.txt', header=None)
_X = np.array([_data[0], _data[1]]).T
_y = np.array(_data[2])
_pos = (_y==1)
plt.scatter(_X[_pos,0], _X[_pos,1], marker='+', c='b')
_neg = (_y==0)
plt.scatter(_X[_neg,0], _X[_neg,1], marker='o', c='y')
plt.legend(['Admitted', 'Not addmitted'], scatterpoints=1)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')

(_Xnorm, _X_mean, _X_std) = Xshaping(_X)
print 'Average=', _X_mean
print 'Std Diviation=', _X_std

_map_degree = 6
_Xnorm = mapFeature(_Xnorm[:,0], _Xnorm[:,1], _map_degree)

print '***Start Fitting***'
#_model = linear_model.LogisticRegression(C=1000000.0)
_model = linear_model.LogisticRegression(C=1.0)
_model.fit(_Xnorm, _y)
print 'Coefficients:', _model.coef_
print 'Intercept:', _model.intercept_
print 'R2:', _model.score(_Xnorm, _y)
print 'J:', CostFunction_logistic(_Xnorm, _y, _model.coef_.reshape(-1,1), _model.intercept_[0])
_accuracy, _precision, _recall, _F_score = evaluate_theta_logistics(_Xnorm, _y, _model.predict(_Xnorm))
print 'accuracy=' +str(np.round(_accuracy,3))
print 'precision=' +str(np.round(_precision,3))
print 'recall=' +str(np.round(_recall,3))
print 'F_score=' +str(np.round(_F_score,3))

_xp = np.linspace(-1.3, 1.3,500)
_yp = np.linspace(-1.3, 1.3,500)
_Xmg, _Ymg = np.meshgrid(_xp, _yp)
_XX = mapFeature( (_Xmg.ravel().T - _X_mean[0,0])/_X_std[0,0], (_Ymg.ravel().T - _X_mean[0,1])/_X_std[0,1], _map_degree)
_Z = np.dot( _XX, _model.coef_.reshape(-1,1) ) + _model.intercept_[0]
_Z = _Z.reshape(_Xmg.shape)
plt.contour(_Xmg, _Ymg, _Z, levels=[0.0], label='Decision Boundary')
#plt.contourf(_Xmg, _Ymg, _Z)
_y0_list = np.where(_y==0)
_y1_list = np.where(_y==1)
plt.scatter(_X[_y0_list,0], _X[_y0_list,1], label='Addmitted', color='red')
plt.scatter(_X[_y1_list,0], _X[_y1_list,1], label='Not addmitted', color='blue')

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show ()
