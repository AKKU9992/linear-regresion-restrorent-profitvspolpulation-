
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"shape of init w is : {w_init.shape}, b init type is {type(b_init)}")

def y_pred(x,y,w,b):
  p=np.dot(w,x)+b
  return p

def compute_cost(x,y,w,b):
  costsum=0.0
  m = x.shape[0]
  for  i in range(m):
    fy=np.dot(x[i],w)+b
    costsum=costsum + (fy-y[i])**2
  total_cost = (1/(2*m))*costsum
  return total_cost

def compute_gradient(x,y,w,b):
  m,n = x.shape
  dy_dw=np.zeros((n,))
  dy_db=0.0

  for  i in range(m):
    fy=(np.dot(x[i],w)+b )-y[i]
    for j in range(n):
      dy_dw[j] = dy_dw[j] + fy*x[i,j]
    dy_db = dy_db + fy
  dy_dw = dy_dw /m
  dy_db = dy_db /m
  return dy_dw,dy_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters, compute_cost, compute_gradient):
  w= copy.deepcopy(w_in)

  b=b_in
  costhistory=[]
  para=[]
  for i in range(num_iters):
    dy_dw,dy_db=compute_gradient(x,y,w,b)
    w=w-alpha*dy_dw
    b=b-alpha*dy_db
    if i<100000:
      costhistory.append(compute_cost(x,y,w,b))
      para.append([w,b])
    if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {costhistory[-1]:8.2f} ")

  return w, b, costhistory,para #return final w,b and J history for graphing

initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 10000
alpha =0.001
mu     = np.mean(x_train, axis=0)
sigma  = np.std(x_train, axis=0)
x_train_scaled = (x_train - mu) / sigma
w_final,b_final, J_history,para = gradient_descent(x_train_scaled ,y_train, initial_w, initial_b, alpha, iterations, compute_cost, compute_gradient)
print(f"the final value of w and b are: ({w_final},{b_final:8.4f})")



plt.plot(J_history)
plt.xlabel("iterations")
plt.ylabel("cost")
plt.title("Cost vs. Iterations")
plt.show()