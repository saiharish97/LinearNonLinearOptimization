# -*- coding: utf-8 -*-
"""Armijos.ipynb

Automatically generated by Colaboratory.

"""

import numpy as np
import pandas as pd
import matplotlib as mp
import math

class ArmijosRule():
    
    # f=lambda self, x:(x[0]*x[0] + 3*x[0]*x[1]+12)
    # dfdx1=lambda self, x: (2*x[0] + 3*x[1])
    # dfdx2=lambda self, x: (3*x[0])


    # f=lambda self, x:(x[0]*x[0] + x[1]*x[1])
    # dfdx1=lambda self, x: (2*x[0])
    # dfdx2=lambda self, x: (2*x[1])

    f=lambda self, x:(-math.log(1-x[0]-x[1])-math.log(x[0])-math.log(x[1]))
    dfdx1=lambda self, x: ((1/(1-x[0]-x[1]))-(1/x[0]))
    dfdx2=lambda self, x: ((1/(1-x[0]-x[1]))-(1/x[1]))



    l2_norm=lambda self, x: (math.sqrt(x[0]*x[0] + x[1]*x[1]))
    
    alpha_0=0.5
    beta=0.1
    sigma=0.1
    count=1

    def __init__(self, alpha_0,beta,sigma):
        self.alpha_0=alpha_0
        self.beta=beta
        self.sigma=sigma

    def backtrackingGD(self, x_k):
        print("Iteration:" + str(self.count))
        self.count+=1
        
        print("Current x:" + str(x_k))

        gradf=np.array([self.dfdx1(x_k),self.dfdx2(x_k)])
        norm_gradf=self.l2_norm(gradf)
        dk= -1*gradf/norm_gradf

        print("Gradient at Current x:" + str(gradf))
        print("Norm of Gradient at Current x:" + str(norm_gradf))
        print("Direction at Current x:" + str(dk))

        if(norm_gradf < 1e-5):
            print("Optimal Solution at x:" + str(x_k))
            return x_k

        alpha=self.alpha_0
        alpha_k=alpha
        if (self.f(x_k + alpha*dk) > self.f(x_k) + self.sigma*alpha*np.dot(gradf,dk)):
            alpha=self.beta * alpha
            while(self.f(x_k + alpha*dk) > self.f(x_k) + self.sigma*alpha*np.dot(gradf,dk)):
                alpha=self.beta * alpha       
        alpha_k=alpha
        x_k1=x_k+alpha_k*dk
        self.backtrackingGD(x_k1)

ar=ArmijosRule(0.2,0.2,0.01)
print(ar.backtrackingGD(np.array([0.00000001,0.00001])))
