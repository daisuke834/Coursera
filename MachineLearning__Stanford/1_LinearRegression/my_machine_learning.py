#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt
import math


def sigmoid(_z):
	return 1.0/ (1.0 + np.exp(-_z))

def CostFunction_linear(_X, _y, _theta, _lambda):
	_M = len(_y)
	_N = len(_theta)-1
	_thetatemp = np.array(_theta).reshape(_N+1,1)
	_hypo = _X.dot(_thetatemp)
	_delta = _hypo - _y
	_theta2 = np.array(_thetatemp)
	_theta2[0] = 0
	_J =  ( np.square(_delta).sum() + _lambda * np.square(_theta2).sum() )/ 2.0 / _M
	_grad = ( np.dot(_X.T, _delta) + _lambda * _theta2) / _M
	return _J, _grad

def CostFunction_logistic(_X, _y, _theta, _lambda):
	_M = len(_y)
	_N = len(_theta)-1
	_thetatemp = np.array(_theta).reshape(_N+1,1)
	_hypo = sigmoid(_X.dot(_thetatemp))
	_theta2 = np.array(_thetatemp)
	_theta2[0] = 0
	_J =  (- np.dot(_y.T, np.log(_hypo)) - np.dot(1-_y.T, np.log(1-_hypo)) + _lambda /2.0 * np.square(_theta2).sum()).sum()/_M
	_grad = ( np.dot(_X.T, _hypo - _y) + _lambda * _theta2) / _M
	_grad = _grad.reshape(len(_grad),1)
	return _J, _grad

def gradientDescent_linear(_X, _y, _theta, _alpha, _num_iters, _lambda):
	_index = 0
	_M = len(_y)
	_J_history = np.zeros(_num_iters)
	while _index < _num_iters:
		_J, _grad = CostFunction_linear(_X, _y, _theta, _lambda)
		_theta = _theta - _alpha * _grad
		_J_history[_index] = _J
		_index = _index +1
	return (_theta, _J_history)

def gradientDescent_logistic(_X, _y, _theta, _alpha, _num_iters, _lambda):
	_index = 0
	_M = len(_y)
	_J_history = np.zeros(_num_iters)
	while _index < _num_iters:
		_J, _grad = CostFunction_logistic(_X, _y, _theta, _lambda)
		_theta = _theta - _alpha * _grad
		_J_history[_index] = _J
		_index = _index +1
	return (_theta, _J_history)

def X_original(_X, _X_mean, _X_std):
	_M = len(_X[:,0])
	_Xtemp = _X[:,1:] * _X_std + _X_mean
	_Xtemp = np.column_stack((np.ones((_M,1)), _Xtemp))
	return _Xtemp

def predict_linear(_X, _theta, _X_mean, _X_std, _original):
	_M = len(_X[:,0])
	_Xtemp = np.array(_X)
	if _original==True:
		if _X_mean is not None and _X_std is not None:
			_Xtemp = (_Xtemp[:,1:] - _X_mean)/ np.array(_X_std)
			_Xtemp = np.column_stack((np.ones((_M,1)), _Xtemp))
	_result = _Xtemp.dot(_theta)
	if len(_result)==1:
		_result = _result[0,0]
	return _result

def predict_logistic(_X, _theta, _X_mean, _X_std, _original):
	return sigmoid( predict_linear(_X, _theta, _X_mean, _X_std, _original) )

def Xshaping(_X, addx0=True, norm=False):
	_N = len(_X[0,:])
	_M = len(_X[:,0])
	_X_mean = None
	_X_std = None
	if norm:
		_X_mean = np.mean(_X, axis=0).reshape(1, _N)
		_X_std = np.std(_X, axis=0).reshape(1, _N)
		_Xnorm = ( _X - _X_mean ) / _X_std
	if addx0:
		_Xnorm = np.column_stack((np.ones((_M,1)), _Xnorm))
	return _Xnorm, _X_mean, _X_std

def mapFeature(_X1, _X2, _degree):
	_degree = 6
	_X = np.ones((len(_X1),1))
	_X1 = _X1.reshape(len(_X1),1)
	_X2 = _X2.reshape(len(_X2),1)
	_i=1
	_j=0
	while _i<=_degree:
		_j=0
		while _j<=_i:
			_X = np.hstack((_X, _X1**(_i-_j) * _X2**_j))
			_j=_j+1
		_i=_i+1
	return _X

def evaluate_theta_logistics(_X, _y, _theta, _threshold):
	_M = len(_y)
	_predict = predict_logistic(_X, _theta, None, None, False)
	_predictY = np.zeros((_M,1))
	_predictY[_predict >= _threshold] = 1
	
	_judge = np.zeros((_M,1))
	_judge[_y == _predictY] = 1
	_accuracy = float(_judge.sum())/float(_M)
	_precision = float((_y * _predictY).sum())/ float(_predictY.sum())
	_recall = float((_y * _predictY).sum())/ float(_y.sum())
	_F_score = 2.0 * _precision * _recall / (_precision + _recall)
	
	return _accuracy, _precision, _recall, _F_score

