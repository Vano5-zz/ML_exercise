import sys
import scipy.optimize, scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotData(data):
    pozitive= data[data[:,2]==1]
    negative= data[data[:,2]==0]
    
    plt.xlabel("Exam 1 score") 
    plt.ylabel("Exam 2 score") 
    plt.xlim([25, 115]) 
    plt.ylim([25, 115])
    
    plt.scatter( negative[:, 0], negative[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted" ) 
    plt.scatter( pozitive[:, 0], pozitive[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted" ) 
    plt.legend() 
    
def sigmoid(z): 
    return scipy.special.expit(z)


def costFunction(theta,x,y):
    m=len(y)
    H_x=sigmoid(x.dot(theta))
    Cost_y_1=np.log(H_x).T.dot(-y)
    Cost_y_0=np.log(1-H_x).T.dot(1-y)
    CostFunction=(Cost_y_1-Cost_y_0)/m
    return CostFunction
def GradientDescen(theta,x,y):
    m=np.shape(x)[0]
    gradient_D=x.T.dot(sigmoid(x.dot(theta))-y)
    return gradient_D
def findMinTheta(theta,x,y):
    result=scipy.optimize.fmin(costFunction,x0=theta,args=(x,y),maxiter=500,full_output=True )
    return result[0],result[1]
    
def plotBound(data,x,theta):
    plotData(data)
    plot_x=np.array(min(x[:,1]),max(x[:,1]))
    plot_y=(-1.0/theta[2])*(theta[1]*plot_x+theta[0])
    plt.plot(plot_x,plot_y)
    plt.show()
def predict (theta,x,binary=True):
    pr=sigmoid(theta.dot(x))
    if binary:
        return 1 if pr>0.5 else 0.
    else:
        return pr
    
def part_1():
    
    data=scipy.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex2\ex2\ex2data1.txt',delimiter=',')
    m,n = np.shape(data)[0],np.shape(data)[1]-1
    x=np.c_[np.ones((m,1)),data[:,:n]]
    y=data[:,n:n+1]
    plotData(data)
    plt.show()

def part_2():
    data=scipy.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex2\ex2\ex2data1.txt',delimiter=',')
    m,n = np.shape(data)[0],np.shape(data)[1]-1
    x=np.c_[np.ones((m,1)),data[:,:n]]   
    theta=np.zeros((n+1,1)) 
    y=data[:,n:n+1]
    
    print costFunction(theta,x,y)
    theta,cost   = findMinTheta(theta,x,y)
    plotBound(data,x,theta)
    plt.show()
    print theta,cost
    test=np.array([1,45,85])
    print predict(test,theta)
    
def main(): 
    np.set_printoptions(precision=6, linewidth=200) 
    part_1() 
    part_2() 


if __name__=='__main__':
    main()