import numpy as np

def f(x):
    x1, x2 = x
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def grad_f(x):
    x1, x2 = x
    return np.array([
        400 * x1**3 - 400 * x1 * x2 + 2 * x1 - 2,
        200 * (x2 - x1**2)
    ])

def hessian_f(x):
    x1, x2 = x
    return np.array([
        [1200 * x1**2 - 400 * x2 + 2, -400 * x1],
        [-400 * x1, 200]
    ])

def inverse_of_regularised_hessian(A,epsilon = 0.01):
    try:
        Ainv = np.linalg.inv(A)
    except:
        # find inverse of A + epsilon * I
        I = np.ones(A.shape)
        Ainv = np.linalg.inv(A + epsilon * I)
    return Ainv
    

def find_best_step_size(f,gradf,x,d,alpha,beta,sigma):
    found_alpha = False
    while not found_alpha:
        if f(x + alpha * d) <= f(x) + sigma * alpha * np.dot(gradf(x),d):
            found_alpha = True
        else:
            alpha = beta * alpha
    return alpha


def gradient_descent_newton(f,x0,gradf,hessian,
                            convergence_thresh=0.12,
                            alpha0=1,
                            beta=0.8,
                            sigma=0.6,
                            max_iters=10**5,
                            print_progress=True):
    """
    @f is the function to be minimized
    @gradf is the gradient of the function to be minimized
    @x0 is the initial guess
    @convergence_thresh is the threshold used to determine convergence
    @max_iters is the maximum number of iteration we try before giving up
    @print_progress flag to indicate whether to print the progress of the algorithm 
    @beta, @sigma are parameters to the optimizer algorithm that finds best step size.
    """
    converged = False
    num_iters_so_far = 0
    x = x0
    trajectory = []
    alpha = alpha0 # initial guess of step size
    while not converged:
        # find descent direction
        d = - gradf(x) / np.linalg.norm(gradf(x))
        
        # choose step size
        alpha=alpha0
        alpha = find_best_step_size(f,gradf,x,d,alpha,beta,sigma) 
        
        # compute regularized hessian
        B = inverse_of_regularised_hessian(hessian(x))
        
        # update x
        num_iters_so_far += 1
        trajectory.append(x)
        x = x + alpha * B@d
        
        # check convergence
        grad_f_norm = np.linalg.norm(gradf(x))
        if grad_f_norm <= convergence_thresh:
            converged = True 
            trajectory.append(x)
            print("\nConverged! The minimial solution occurs @x = ", x, "\nwith value f(x)=",f(x), "\nand ||gradf||=",grad_f_norm)
        if not converged and num_iters_so_far > max_iters:
            converged = True
            print("Failed to converge :(")
            
        # output progress
        if print_progress and num_iters_so_far % 10 == 0:
            print("\nIteration: ", num_iters_so_far,"\nx: ",x, " f(x)", f(x), "\n||gradf||:",grad_f_norm) 
            print("prev_x: ", trajectory[-1], " f(prev_x): ",f(trajectory[-1]))
            print("B: ",B)
            print("--------------------------------------------------------")
    return x,np.array(trajectory) 

x = np.array([0.5, -0.5])
x_opt,trajectory = gradient_descent_newton(f,x,grad_f,hessian_f,max_iters=10)

for t in range(len(trajectory)):
    print("iteration ",t," : ",trajectory[t])

#OP:
# Iteration:  10 
# x:  [ 0.50005204 -0.48832404]  f(x) 54.769871622520185 
# ||gradf||: 208.14929304835024
# prev_x:  [ 0.50004676 -0.48949171]  f(prev_x):  54.94166849979957
# B:  [[0.00335778 0.0033581 ]
#  [0.0033581  0.00835841]]
# --------------------------------------------------------
# Failed to converge :(
# iteration  0  :  [ 0.5 -0.5]
# iteration  1  :  [ 0.50000513 -0.49883248]
# iteration  2  :  [ 0.50001028 -0.49766494]
# iteration  3  :  [ 0.50001544 -0.49649738]
# iteration  4  :  [ 0.50002062 -0.49532981]
# iteration  5  :  [ 0.50002582 -0.49416222]
# iteration  6  :  [ 0.50003103 -0.49299462]
# iteration  7  :  [ 0.50003626 -0.491827  ]
# iteration  8  :  [ 0.5000415  -0.49065936]
# iteration  9  :  [ 0.50004676 -0.48949171]
# iteration  10  :  [ 0.50005204 -0.48832404]
