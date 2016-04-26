#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as optimize
import my_machine_learning as _myml

_fh = open('ex2data1.txt')
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

_X_mean = None
_X_std = None

(_X, _X_mean, _X_std) = _myml.Xshaping(_X, addx0=True, norm=True)
print 'Average=', _X_mean
print 'Std Diviation=', _X_std

_initial_theta = np.zeros((_N+1,1))
_theta = np.array(_initial_theta)

_J, _grad = _myml.CostFunction_logistic(_X, _y, _theta, 0.0)
print "Initial J=\t", _J
print "Initial Grad=\t", _grad.T

_lambda = 0.0

# Optimization
print '***Optimization***'
_f = lambda _t: _myml.CostFunction_logistic(_X, _y, _t, _lambda)[0]
_f_derivative = lambda _t: _myml.CostFunction_logistic(_X, _y, _t, _lambda)[1].ravel()
_opt_result = optimize.minimize(_f, _initial_theta.ravel(), method='BFGS', jac=_f_derivative, options={'disp':True})
_theta = _opt_result.x.reshape(_N+1,1)

# Result
print '***Result***'
_J, _grad = _myml.CostFunction_logistic(_X, _y, _theta, 0.0)
print 'J=', _J
_accuracy, _precision, _recall, _F_score = _myml.evaluate_theta_logistics(_X, _y, _theta, 0.5)
print 'accuracy=' +str(np.round(_accuracy,3))
print 'precision=' +str(np.round(_precision,3))
print 'recall=' +str(np.round(_recall,3))
print 'F_score=' +str(np.round(_F_score,3))
_Xoriginal = _myml.X_original(_X, _X_mean, _X_std)
plt.scatter((_Xoriginal[:,1])[_y.ravel()==0], (_Xoriginal[:,2])[_y.ravel()==0], label='Admitted', color='red')
plt.scatter((_Xoriginal[:,1])[_y.ravel()==1], (_Xoriginal[:,2])[_y.ravel()==1], label='Not addmitted', color='blue')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
print 'theta = ', _theta.T
_Xtemp = np.ones((_M, _N+1))
_Xtemp[:,1] = np.linspace(_X[:,1].min(), _X[:,1].max(),100)
_Xtemp[:,2] = - (_theta[0,0] + _theta[1,0] * _Xtemp[:,1]) / _theta[2,0]
_Xtemp = _myml.X_original(_Xtemp, _X_mean, _X_std)
plt.plot(_Xtemp[:,1], _Xtemp[:,2], label='Decision Boundary')
plt.title('Logistic Regression: BFGS by scipy, lambda='+str(round(_lambda,5)))
plt.legend()
plt.show ()
