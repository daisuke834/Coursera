#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt
import math
import my_machine_learning as _myml

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

_theta = np.zeros((_N+1,1))

_J, _grad = _myml.CostFunction_linear(_X, _y, _theta, 0.0)
print "J=", _J

_num_iters = 400
_lambda = 0.0
_alphaarray = [1.0, 0.3, 0.1, 0.03, 0.01]
_J_Min = None
_alpha_JMIN = None
for _alpha in _alphaarray:
	_theta = np.zeros((_N+1,1))
	(_theta, _J_history) = _myml.gradientDescent_linear(_X, _y, _theta, _alpha, _num_iters, _lambda)
	print 'alpha=', _alpha, ', lambda=', _lambda
	_J, _grad = _myml.CostFunction_linear(_X, _y, _theta, _lambda)
	print '\tcost=', _J
	print '\tfor [1650, 3], we predict $', np.round(_myml.predict_linear(np.array([[1, 1650, 3]]), _theta, _X_mean, _X_std, True), 2)
	plt.plot(range(_num_iters), _J_history, label='alpha='+str(_alpha)+', lambda='+str(_lambda))
	if _J_Min is None or _J<_J_Min or math.isnan(_J_Min):
		_J_Min  = _J
		_alpha_JMIN = _alpha

plt.xlabel('Iteration')
plt.ylabel('Cost Function')
plt.legend()
plt.yscale('log')
plt.ylim([0,1000000000000])
#plt.yticks(range(0,8000000, 1000000))
plt.show ()

print '***Final***'
_theta = np.zeros((_N+1,1))
_alpha = _alpha_JMIN
(_theta, _J_history) = _myml.gradientDescent_linear(_X, _y, _theta, _alpha, _num_iters, _lambda)
_J, _grad = _myml.CostFunction_linear(_X, _y, _theta, _lambda)
print 'alpha=', _alpha
print 'J=', _J
print 'theta=:', _theta.T 
print 'for [1650, 3], we predict $', np.round(_myml.predict_linear(np.array([[1, 1650, 3]]), _theta, _X_mean, _X_std, True), 2)

