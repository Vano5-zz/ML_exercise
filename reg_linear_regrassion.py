import sys
import scipy.optimize, scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.plotly as py




data=scipy.genfromtxt('F:\EPAM\coursera\ML_COURSERA_GARVARD\machine-learning-ex2\ex2\ex2data2.txt',delimiter=',')

def plotData(data):
    
    pozitive=data[data[:,2]==1]
    negative=data[data[:,2]==0]
    plt.scatter( negative[:, 0], negative[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted" ) 
    plt.scatter( pozitive[:, 0], pozitive[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted" ) 
    plt.legend() 
    

def segmoid (z):
    return scipy.special.expit(z)
    
          
print    plotData(data) 