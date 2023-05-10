import numpy as np

def function_m(x):
    return (x[0] - 4)**2 + (2 * x[1] - 3)**2

def gradient_m(x):
    df_dx1 = 2 * (x[0] - 4)
    df_dx2 = 4 * (2 * x[1] - 3)
    return np.array([df_dx1, df_dx2])

def hessianm_m(x):
    return np.array([[2, 0],
                     [0, 8]])

def function_barrier(x):
    return np.log(5 - x[0] - x[1])

def gradient_barrier(x):
    df_dx1 = -1 / (5 - x[0] - x[1])
    df_dx2 = -1 / (5 - x[0] - x[1])
    return np.array([df_dx1, df_dx2])

def hessian_barrier(x):
    denom = (5 - x[0] - x[1])**2
    return np.array([[1/denom, 1/denom],
                     [1/denom, 1/denom]])


def inverse_of_regularised_hessian(A,epsilon = 0.01):
    try:
        Ainv = np.linalg.inv(A)
    except:
        # find inverse of A + epsilon * I
        I = np.ones(A.shape)
        Ainv = np.linalg.inv(A + epsilon * I)
    return Ainv
    

def find_best_step_size(x,d,alpha,beta,sigma,k):
    found_alpha = False
    gradfx=gradient_m(x)+(1/(k+1))*gradient_barrier(x)
    while not found_alpha:
        if function_m(x + alpha * d) + function_barrier(x + alpha * d) <= function_m(x) + (1/(k+1))*function_barrier(x) + sigma * alpha * np.dot(gradfx,d):
            found_alpha = True
        else:
            alpha = beta * alpha
    return alpha


def gradient_descent_newton(x0,
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
        gradfx=gradient_m(x)+(1/(num_iters_so_far+1))*gradient_barrier(x)
        d = - gradfx / np.linalg.norm(gradfx)
        
        # choose step size
        alpha=alpha0
        alpha = find_best_step_size(x,d,alpha,beta,sigma,k=num_iters_so_far) 
        
        # compute regularized hessian
        B = inverse_of_regularised_hessian(hessianm_m(x)+(1/(num_iters_so_far+1))*(hessian_barrier(x)))
        
        # update x
        num_iters_so_far += 1
        trajectory.append(x)
        x_new = x + alpha * B@d
        
        # check convergence
        grad_f_norm = np.linalg.norm((x_new-x))
        x=x_new
        if grad_f_norm <= convergence_thresh:
            converged = True 
            trajectory.append(x)
            print("\nConverged! The minimial solution occurs @x = ", x, "\nwith value f(x)=",function_m(x), "\nand ||x_new - x||=",grad_f_norm)
        if not converged and num_iters_so_far > max_iters:
            converged = True
            print("Failed to converge :(")
            
        # output progress
        if print_progress and num_iters_so_far % 10 == 0:
            print("\nIteration: ", num_iters_so_far,"\nx: ",x, " f(x)", function_m(x), "\n||gradf||:",grad_f_norm) 
            print("prev_x: ", trajectory[-1], " f(prev_x): ",function_m(trajectory[-1]))
            print("B: ",B)
            print("--------------------------------------------------------")
    return x,np.array(trajectory) 

x = np.array([0,0])
x_opt,trajectory = gradient_descent_newton(x,max_iters=10,convergence_thresh=0.000001)

for t in range(len(trajectory)):
    print("iteration ",t," : ",trajectory[t])



# OP:
# iteration  0  :  [0. 0.]
# iteration  1  :  [0.27145144 0.10187725]
# iteration  2  :  [0.54533084 0.20462885]
# iteration  3  :  [0.81993557 0.30764067]
# iteration  4  :  [1.09480564 0.41074731]
# iteration  5  :  [1.3697195  0.51386952]
# iteration  6  :  [1.64451601 0.61695022]
# iteration  7  :  [1.91902849 0.71993111]
# iteration  8  :  [2.1930318  0.82273459]
# iteration  9  :  [2.46615845 0.92523582]
# iteration  10  :  [2.73770756 1.02720084]