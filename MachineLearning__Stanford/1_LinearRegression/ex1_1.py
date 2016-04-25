#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt
import my_machine_learning as _myml

_fh = open('ex1data1.txt')
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

#plt.scatter(_X[:,0], _y)
#plt.xlabel('Population of City in 10,000s')
#plt.ylabel('Profit in $10,000s')
#plt.show ()

(_X, _X_mean, _X_std) = _myml.Xshaping(_X, addx0=True, norm=False)

_theta = np.zeros((_N+1,1))
_J, _grad = _myml.CostFunction_linear(_X, _y, _theta, 0.0)
print "J=", _J

_alpha = 0.01
_num_iters = 1500
(_theta, _J_history) = _myml.gradientDescent_linear(_X, _y, _theta, _alpha, _num_iters, 0.0)
print '***theta***'
print _theta

plt.scatter(_X[:,1], _y, label='Training data')
plt.plot(_X[:,1], _myml.predict_linear(_X, _theta, _X_mean, _X_std, False), label='Linear regression')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show ()
print "Predict: For population = 35,000, we predict a profit of", _myml.predict_linear(np.array([[1.0, 3.5]]), _theta, _X_mean, _X_std, True) * 10000.0
print "Predict: For population = 70,000, we predict a profit of", _myml.predict_linear(np.array([[1.0, 7.0]]), _theta, _X_mean, _X_std, True) * 10000.0

plt.plot(range(1500), _J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost Function')
plt.show ()
