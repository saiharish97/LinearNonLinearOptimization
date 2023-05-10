import numpy as np

def f(x):
    term1 = np.sqrt(np.exp(x[0]) + np.exp(-x[1]))
    term2 = np.log(np.exp(x[1]) + np.exp(x[2]))
    return term1 + term2

def grad_f(x):
    term1 = np.array([
        0.5 * np.exp(x[0]) / np.sqrt(np.exp(x[0]) + np.exp(-x[1])),
        -0.5 * np.exp(-x[1]) / np.sqrt(np.exp(x[0]) + np.exp(-x[1])),
        0.0
    ])
    term2 = np.array([
        0.0,
        np.exp(x[1]) / (np.exp(x[1]) + np.exp(x[2])),
        np.exp(x[2]) / (np.exp(x[1]) + np.exp(x[2]))
    ])
    return term1 + term2


#closest point on plane to the given point
def projection(x):
    y2=(2*x[0]+10*x[1]+6*x[2]-4)/14
    y3=(-3*x[0]+6*x[1]+5*x[2]+6)/14
    y1=2+2*y2-3*y3
    return(np.array([y1,y2,y3]))


def gradient_projection(x, alpha, iter=5):
    trajectory=[]
    trajectory.append(x)
    while iter>0:
        x_old=x
        x_new_out=x-alpha*grad_f(x)
        x_new_proj=projection(x_new_out)
        trajectory.append(x_new_proj)
        x=x_new_proj
        if(np.linalg.norm(x_new_proj-x_old) < 0.0000001):
            return x,trajectory
        iter-=1
    if iter==0:
        return x,trajectory

# print(f(np.array([3,2,1])))

x_opt,trajectory=gradient_projection(np.array([3,2,1]),alpha=0.1,iter=5)

for t in range(len(trajectory)):
    print("iteration ",t," : ",trajectory[t])

# OP:
# iteration  0  :  [3 2 1]
# iteration  1  :  [2.78815338 1.90542557 1.00756592]
# iteration  2  :  [2.59815459 1.81491633 1.01055936]
# iteration  3  :  [2.42604225 1.72786817 1.00989803]
# iteration  4  :  [2.26888276 1.64383872 1.00626489]
# iteration  5  :  [2.12444111 1.56249605 1.00018366]