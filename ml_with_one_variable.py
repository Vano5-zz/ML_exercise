import sys
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data=sp.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex1\machine-learning-ex1\ex1\ex1data1.txt',delimiter=',')
x=data[:,0]
y=data[:,1]
m=len(y)

###############################
##Plot Graf
    #plt.scatter(x,y,s=10)
    #plt.title ('Plot Graf')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.autoscale(tight=True)
    #plt.grid(True)
    #plt.show()
##################################    
#for xi in x:
#    x1=np.append(x1,[1,xi])
#x=x1.reshape(m,2)
##################################
#Polyfit
#fpl=sp.polyfit(x,y,1,True)
#x_fpl=fpl[:1]
#y_fpl=fpl[1:2]
#print x_fpl,y_fpl
##################################

theta=np.zeros((2,1))
iterations = 81;
alpha = 0.01;
def computerCost(x,y,theta):
    J=0
    m=len(y)
    predict = x.dot(theta)
    sqError=predict-y
#sum(sqError**2)=sqError.T.dot(sqError)   
    J=(1.0/(2*m))*sqError.T.dot(sqError) 
    return J
def Ggradient_Descent(x,y,theta,alpha,iterations):
    m=y.size
    J_history=np.zeros(shape=(iterations,1))
    for i in range(iterations):
        predict=x.dot(theta)
        theta_size=theta.size
        
        for th_i in range(theta_size):
             temp=x[:,th_i]
             temp.shape=(m,1)
             error_x=(predict-y)*temp
             theta[th_i][0]=theta[th_i][0]-alpha*(1.0/m)*error_x.sum()
        J_history=computerCost(x,y,theta)
        #J_history.shape=(iterations,1)
    
    return theta   
    
def plot(x,y):
    plt.plot(x,y,'rx',markersize=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot_1():
    data=sp.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex1\machine-learning-ex1\ex1\ex1data1.txt',delimiter=',')
    x=data[:,0]
    y=data[:,1] 
    m=len(y)
    y=y.reshape(m,1)
    
    plot(x,y)
    plt.show(block=True) 
    
def plot_2():
    data=sp.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex1\machine-learning-ex1\ex1\ex1data1.txt',delimiter=',')
    x=data[:,0]
    y=data[:,1] 
    m=len(y)
    y=y.reshape(m,1)
    x1=np.array([])
    
    for xi in x:
        x1=np.append(x1,[1,xi])
    x=x1.reshape(m,2)
    
    theta=np.zeros((2,1))
    iterations = 1500;
    alpha = 0.01;
   
    cost=computerCost(x,y,theta)
    theta=Ggradient_Descent(x,y,theta,alpha,iterations)
    
    print cost
    print theta 
    pr1=np.array([1,3.5]).dot(theta)
    pr2=np.array([1,7]).dot(theta)
   
    print pr1
    print pr2
    y_1=x.dot(theta)
    y_1.shape=(m,1)
    plt.title('Linear regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x[:,1],y,'b-')
    plt.plot(x[:,1],y_1,'r-')
    plt.show(block=True)
        
print plot_2()

def plot_3():
    data=sp.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex1\machine-learning-ex1\ex1\ex1data1.txt',delimiter=',')
    x=data[:,0]
    y=data[:,1] 
    m=len(y)
    y=y.reshape(m,1)
    
    x1=np.array([])
    
    for xi in x:
        x1=np.append(x1,[1,xi])
    x=x1.reshape(m,2)
    
    theta=np.zeros((2,1))
    
    
    theta0_vals = np.linspace(-10., 10., 100);
    theta1_vals = np.linspace(-1., 4., 100);
   
    J_vals=np.zeros((len(theta0_vals),len(theta1_vals)))
    
    for i,v0 in enumerate(theta0_vals):
        for j,v1 in enumerate (theta1_vals):
            theta=np.array((theta0_vals[i],theta1_vals[j])).reshape(2,1)
            J_vals[i][j]=computerCost(x,y,theta)
    
    R, P = np.meshgrid(theta0_vals,theta1_vals) 
    #plt.xlabel('Q1')
    #plt.ylabel('Q2')
    #plt.zlabel('J(Q)')
    fig = plt.figure() 
    #ax=fig.add_subplot(R,P,J_vals,projection='3d')
    ax 	=plt.gca(projection='3d') 
    ax.plot_surface(R, P, J_vals) 
    plt.show(block=True) 
    #print R,P
    theta=np.zeros((2,1))
    iterations = 1500;
    alpha = 0.01;
    
    #cost=computerCost(x,y,theta)
    theta=Ggradient_Descent(x,y,theta,alpha,iterations)
 
    fig = plt.figure() 
    ax 	= fig.gca(projection='3d') 
    plt.contourf(R, P, J_vals.T, np.logspace(-2, 3, 20)) 
    plt.plot(theta[0], theta[1], 'rx', markersize = 10) 
    plt.show(block=True) 
    print theta[0], theta[1]


def main(): 
    #np.set_printoptions(precision=6, linewidth=200) 
 
 
    plot_1() 
    plot_2() 
    plot_3() 
   

 
    sys.exit() 
 
  
 
if __name__ == '__main__': 
     main() 


    
   