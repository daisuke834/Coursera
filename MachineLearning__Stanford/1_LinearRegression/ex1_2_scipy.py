#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as optimize
import my_machine_learning as _myml

print '*************************'
print '***ex1data2***'
_fh = open('ex1data2.txt')
_data_array = list()
_index =0
_M = None
_N = None

for _line in _fh:
	_line = _line.strip()
	_data = _line.split(',')
	if len(_data)<2: continue;
	if _index == 0: _N = len(_data)-1
	for _element in _data:
		_data_array.append(float(_element))
	_index = _index+1
_M = _index
_data_array = np.array(_data_array)
_data_array = _data_array.reshape(_M,_N+1)

_X = _data_array[:,0:-1].reshape(_M,_N)
_y = _data_array[:,-1].reshape(_M,1)
print '(_M, _N)= (', _M, ', ', _N, ')'

(_X, _X_mean, _X_std) = _myml.Xshaping(_X, addx0=True, norm=True)
print 'Average=', _X_mean
print 'Std Diviation=', _X_std

_initial_theta = np.zeros((_N+1,1))
_theta = np.array(_initial_theta)

_J, _grad = _myml.CostFunction_linear(_X, _y, _theta, 0.0)
print "J=", _J

_lambda = 0.0

# Optimization
print '***Optimization Start**********'
_f = lambda _t: _myml.CostFunction_linear(_X, _y, _t, _lambda)[0]
_f_derivative = lambda _t: _myml.CostFunction_linear(_X, _y, _t, _lambda)[1].ravel()
_opt_result = optimize.minimize(_f, _initial_theta.ravel(), method='BFGS', jac=_f_derivative, options={'disp':True})
_theta = _opt_result.x.reshape(_N+1,1)

# Result
print '***Optimization Result**********'
_J, _grad = _myml.CostFunction_linear(_X, _y, _theta, _lambda)
print 'J=', _J
print 'theta=:', _theta.T 
print 'for [1650, 3], we predict $', np.round(_myml.predict_linear(np.array([[1, 1650, 3]]), _theta, _X_mean, _X_std, True), 2)


