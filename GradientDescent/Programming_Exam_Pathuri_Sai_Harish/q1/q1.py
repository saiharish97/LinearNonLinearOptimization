# Step 1: Coding up blackboxes for f, gradf, hessianf 
import numpy as np

Q=np.array([[1,0],[0,900]])
p=np.array([[100],[0]])

def f(x):
    return (1/2)*(x.transpose() @ Q @ x) - (p.transpose() @ x) 

def g(x):
    return (Q@x - p)

def gradient_descent_with_optimal_step_size_exact(f,x0,gradf,e=0.0001,max_iters=10000):
    """
    @f is the function to be minimized
    @gradf is the gradient of the function to be minimized
    @x0 is the initial guess
    @convergence_thresh is the threshold used to determine convergence
    @max_iters is the maximum number of iteration we try before giving up
    @print_progress flag to indicate whether to print the progress of the algorithm 
    """
    converged = False
    num_iters_so_far = 0
    x = x0
    alpha = 0.2 # initial guess of step size
    trajectory = []
    convergence_thresh=e
    while not converged:
        # find descent direction
        d = - gradf(x) / np.linalg.norm(gradf(x))
        
        alpha = ((p.transpose() @ d) - (0.5)*(d.transpose() @ Q @ x) - (0.5)*(x.transpose() @ Q @ d))/(d.transpose() @ Q @ d)

        # update x
        num_iters_so_far += 1
        trajectory.append(x) 
        x_new = x + alpha * d
        
        # check convergence
        x_norm = np.linalg.norm(x_new-x)
        x=x_new
        if x_norm <= convergence_thresh:
            converged = True
            trajectory.append(x)
            print("\nConverged! The minimial solution occurs @x = ", x, "\nwith value f(x)=",f(x), "\nand ||x_new - x||=",x_norm)
        if not converged and num_iters_so_far > max_iters:
            converged = True
            print("Failed to converge :(")
            
    return x,np.array(trajectory)

x = np.array([[1000],[1]])
x_opt,trajectory = gradient_descent_with_optimal_step_size_exact(f,x,g,e=0.0001)

# for t in range(len(trajectory)):
#     print("iteration ",t," : x=(",trajectory[t][0][0]," , ",trajectory[t][1][0],")^T")

print("Optimal Value of x:(", x_opt[0][0],",",x_opt[1][0],")^T")


# OP:
#Converged! The minimial solution occurs @x =  [[1.00031716e+02]
#  [3.52396900e-05]] 
# with value f(x)= [[-4999.9994965]] 
# and ||x_new - x||= 9.978376600390819e-05
# Optimal Value of x:( 100.03171572102988 , 3.5239690033205575e-05 )^T
#
