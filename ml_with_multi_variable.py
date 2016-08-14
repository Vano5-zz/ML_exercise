import sys
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
##############################

data=sp.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex1\machine-learning-ex1\ex1\ex1data2.txt',delimiter=',')
x=data[:,0:2]
y=data[:,2:3]
#theta=np.zeros((2,1))
m=len(y)
def featureNormalize(data):
    mu =np.mean(data,axis=0)
    data_norm=data-mu
    sigma =np.std(data_norm,axis=0,ddof=1)
    data_norm=data_norm/sigma
    return mu,data_norm,sigma
def featureNormalizeLoop(data):
    mu=[]
    sigma=[]
    data_norm=np.zeros(np.shape(data),data.dtype)
    for col in range(np.shape(data)[1]):
        mu.append(np.mean(data[:,col]))
        sigma.append(np.std(data[:,col],ddof=1))
        data_norm[:,col]=map(lambda x:(x-mu[col])/sigma[col],data[:,col])
    return mu,sigma,data_norm

def costFunction(x,y,theta):
    m=len(y)
    sqlError=(x.dot(theta)-y)
    
    return (1.0/2.0*m)*sqlError.T.dot(sqlError)
 
    
def GradientDiscent(x,y,theta,alpha,iteration):
    m=len(y)
    J_history=[]
    alpha_m=alpha/m
    for i in range(0,iteration):
        pred=x.T.dot(x.dot(theta)-y)
        theta=theta-alpha_m*pred
        J_history.append(costFunction(x,y,theta))
    return  J_history,theta

def normalEquations(x,y):
    return  np.linalg.inv(x.T.dot( x )).dot( x.T ).dot( y ) 

def plot_1():
    data=sp.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex1\machine-learning-ex1\ex1\ex1data2.txt',delimiter=',')
    x=data[:,0:2]
    y=data[:,2:3]
    m=len(y)
    
    mu,x,sigma =featureNormalize(x)
    print mu
    print x
    print sigma
def plot2():
    data=sp.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex1\machine-learning-ex1\ex1\ex1data2.txt',delimiter=',')
    x=data[:,0:2]
    y=data[:,2:3]
    m=len(y)
    
    mu,x,sigma =featureNormalize(x)
    #x1=np.array([])
    #
    #for xi in x:
    #    x1=np.append(x1,[1,xi])
    
    x=np.c_[np.ones((m,1)),x]
    
    iteration 	= 4 
    alphas 	= [0.01] 
    
    for alpha in alphas:
        theta=np.zeros((3,1))
        J_history,theta=GradientDiscent(x,y,theta,alpha,iteration)
        
        number_iteration=np.array( [x for x in range( 1, iteration + 1 )] ).reshape( iteration, 1)
        
  #      plt.plot(number_iteration, J_history, '-b' )
  #      plt.title( "Alpha = %f" % (alpha) ) 
 	#plt.xlabel('Number of iterations') 
 	#plt.ylabel('Cost J') 
 	#plt.xlim( [0, 50] ) 
  #	plt.show( block=True ) 
  	
  	
  #	test = np.array([1.0, 1650.0, 3.0]) 
 	## exclude intercept units 
 	#test[1:] = (test[1:] - mu) / sigma 
 	#print test.dot( theta ) 
 	#print number_iteration
    J_history=np.array(J_history).reshape(iteration,1)	
    print J_history[:3],number_iteration

print plot2()


    
        
    
        
        