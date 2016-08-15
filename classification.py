import sys
import scipy as sp
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
    

 
def part_1():
    
    data=sp.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex2\ex2\ex2data1.txt',delimiter=',')
    
    m,n = np.shape(data)[0],np.shape(data)[1]-1
    
    x=np.c_[np.ones((m,1)),data[:,:n]]
    y=data[:,n:n+1]
    
    plotData(data)
    plt.show()
print part_1()
    
    