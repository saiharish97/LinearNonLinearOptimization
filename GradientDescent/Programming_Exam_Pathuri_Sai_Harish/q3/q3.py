import numpy as np
import math

def f(x):
    x1, x2 = x
    return 10 * x1**4 - 20 * x1**2 * x2 + 10 * x2**2 - 2 * x1 + 5

def grad(x):
    x1, x2 = x
    return np.array([
        40 * x1**3 - 40 * x1 * x2 - 2,
        -20 * x1**2 + 20 * x2
    ])


def find_best_step_size(f,gradf,x,d,alpha,beta,sigma):
    found_alpha = False
    while not found_alpha:
        if f(x + alpha * d) <= f(x) + sigma * alpha * np.dot(gradf(x),d):
            found_alpha = True
        else:
            alpha = beta * alpha
    return alpha

def conjugate_gradient(x,e,btalpha,btbeta,btsigma):
    d=-grad(x)
    trajectory=[]
    trajectory.append(x)
    while True:
        alpha=find_best_step_size(f,grad,x,d,btalpha,btbeta,btsigma)
        x_new=x+alpha*d
        trajectory.append(x_new)
        grad_x_k_1=grad(x_new)
        grad_x_k=grad(x)
        beta= (grad_x_k_1.transpose() @ grad_x_k_1)/(grad_x_k.transpose() @ grad_x_k)
        d= -grad_x_k_1 + beta*d
        diff_x_norm = np.linalg.norm(x_new - x)
        x=x_new
        if(diff_x_norm<e):
            print("\nConverged! The minimial solution occurs @x = ", x, "\nwith value f(x)=",f(x), "\nand ||x_new - x||=",diff_x_norm)
            return x_new,trajectory
        

x_opt,trajectory=conjugate_gradient(np.array([-1.5,3]),0.01,1,0.8,0.6)

for t in range(len(trajectory)):
    print("iteration ",t," : ",trajectory[t])

# OP:
#     Converged! The minimial solution occurs @x =  [-1.70108644  2.92845021] 
# with value f(x)= 8.41425207184802 
# and ||x_new - x||= 0.009945301705856292
# iteration  0  :  [-1.5  3. ]
# iteration  1  :  [-1.66244941  2.9433316 ]
# iteration  2  :  [-1.69209143  2.93269271]
# iteration  3  :  [-1.70108644  2.92845021]