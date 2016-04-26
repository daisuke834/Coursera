#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn import linear_model
from sklearn import svm

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

print '*************************'
print '***ex2data1***'
_data = pd.read_csv('ex2data1.txt', header=None)
_X = np.array([_data[0], _data[1]]).T
_y = np.array(_data[2])

(_Xnorm, _X_mean, _X_std) = Xshaping(_X)
print 'Average=', _X_mean
print 'Std Diviation=', _X_std

print '***Start Fitting***'
#_kernel = 'linear'
_kernel = 'rbf'
_C = 1000
_model = svm.SVC(C=_C, kernel=_kernel)
_model.fit(_Xnorm, _y)
_accuracy, _precision, _recall, _F_score = evaluate_theta_logistics(_Xnorm, _y, _model.predict(_Xnorm))
print 'accuracy=' +str(np.round(_accuracy,3))
print 'precision=' +str(np.round(_precision,3))
print 'recall=' +str(np.round(_recall,3))
print 'F_score=' +str(np.round(_F_score,3))

_xp = np.linspace(30, 100,500)
_yp = np.linspace(30, 100,500)
_Xmg, _Ymg = np.meshgrid(_xp, _yp)
_XX = np.hstack( ( (_Xmg.ravel().reshape(-1,1) - _X_mean[0,0])/_X_std[0,0], (_Ymg.ravel().reshape(-1,1) - _X_mean[0,1])/_X_std[0,1] ) )
_Z = _model.predict(_XX)
_Z = _Z.reshape(_Xmg.shape)
plt.contour(_Xmg, _Ymg, _Z, levels=[0.5], label='Decision Boundary')
#plt.contourf(_Xmg, _Ymg, _Z)
_y0_list = np.where(_y==0)
_y1_list = np.where(_y==1)
plt.scatter(_X[_y0_list,0], _X[_y0_list,1], label='Admitted', color='red')
plt.scatter(_X[_y1_list,0], _X[_y1_list,1], label='Not addmitted', color='blue')
plt.title('Scikit-learn: SVM, '+_kernel+', C='+str(round(_C,3)))
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show ()
