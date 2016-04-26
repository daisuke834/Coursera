#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt
import math
import my_machine_learning as _myml

_fh = open('ex2data2.txt')
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

_X1 = np.array((_data_array[:,0])).reshape(_M,1)
_X2 = np.array((_data_array[:,1])).reshape(_M,1)
_y = _data_array[:,-1].reshape(_M,1)

_map_degree = 6

_X_mean = None
_X_std = None
_X = _myml.mapFeature(_X1, _X2, _map_degree)
_N = len(_X[0,:]) - 1
print '(_M, _N)= (', _M, ', ', _N, ')'

_initial_theta = np.zeros((_N+1,1))
_theta = np.array(_initial_theta)

_J, _grad = _myml.CostFunction_logistic(_X, _y, _theta, 0.0)
print "Initial J=\t", _J
print "Initial Grad=\t", _grad.T

_num_iters = 20000
_lambda = 0.0
_alphaarray = [18.0, 17.0, 16.0, 15.5, 15.3, 15.0, 14.8, 14.5, 14.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01]
#_alphaarray = [0.3]
_J_Min = None
_alpha_JMIN = None
for _alpha in _alphaarray:
	_theta = _initial_theta
	(_theta, _J_history) = _myml.gradientDescent_logistic(_X, _y, _theta, _alpha, _num_iters, _lambda)
	print 'alpha=', _alpha
	_J, _grad = _myml.CostFunction_logistic(_X, _y, _theta, 0.0)
	print '\tcost=', _J
	_accuracy, _precision, _recall, _F_score = _myml.evaluate_theta_logistics(_X, _y, _theta, 0.5)
	print 'accuracy=' +str(np.round(_accuracy,3)), ', precision=' +str(np.round(_precision,3)), ', recall=' +str(np.round(_recall,3)), ', F_score=' +str(np.round(_F_score,3))
	plt.plot(range(_num_iters), _J_history, label='alpha='+str(_alpha))
	if _J_Min is None or _J<_J_Min or math.isnan(_J_Min):
		_J_Min  = _J
		_alpha_JMIN = _alpha

plt.xlabel('Iteration')
plt.ylabel('Cost Function')
plt.legend()
plt.yscale('log')
#plt.ylim([0,1000000000000])
#plt.yticks(range(0,8000000, 1000000))
plt.show ()

print '***Final***'
_theta = _initial_theta
_alpha = _alpha_JMIN
(_theta, _J_history) = _myml.gradientDescent_logistic(_X, _y, _theta, _alpha, _num_iters, _lambda)
_J, _grad = _myml.CostFunction_logistic(_X, _y, _theta, 0.0)
print 'alpha=', _alpha
print 'J=', _J
_accuracy, _precision, _recall, _F_score = _myml.evaluate_theta_logistics(_X, _y, _theta, 0.5)
print 'accuracy=' +str(np.round(_accuracy,3)), ', precision=' +str(np.round(_precision,3)), ', recall=' +str(np.round(_recall,3)), ', F_score=' +str(np.round(_F_score,3))
print _theta[0,0], _theta[1,0], _theta[2,0]

_xp = np.linspace(-1.3, 1.3,500)
_yp = np.linspace(-1.3, 1.3,500)
_Xmg, _Ymg = np.meshgrid(_xp, _yp)
_XX = _myml.mapFeature(_Xmg.ravel().T, _Ymg.ravel().T, _map_degree)
_Z = _myml.predict_linear( _XX, _theta, _X_mean, _X_std, False).reshape(_Xmg.shape)
plt.contour(_Xmg, _Ymg, _Z, levels=[0.0], label='Decision Boundary')
#plt.contourf(_Xmg, _Ymg, _Z)
_y0_list = np.where(_y==0)
_y1_list = np.where(_y==1)
plt.scatter(_X[_y0_list,1], _X[_y0_list,2], label='PASS', color='red')
plt.scatter(_X[_y1_list,1], _X[_y1_list,2], label='FAIL', color='blue')
plt.title('Polynomial Gradient Descent: Scratch, lambda='+str(_lambda)+', alpha='+str(_alpha))
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.show ()
