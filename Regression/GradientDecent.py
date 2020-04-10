'''
The following rule is applied in the code
mean squared error =1/n*sum(delta(error)^2)
mse = 1/n*sum(y[i]-y_predicted)
mse = 1/n*sum(y[i]-(m*x[i]+b))^2
d/dm = 2/n*sum(-x[i]*(y[i]-(m*x[i]+b)))
d/db = 2/n*sum(y[i]-(m*x[i]+b))


'''
import numpy as np

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

m_curr=b_curr=0
iterations = 10000
n=len(x)
learning_rate = .01
for i in range(iterations):
    y_predicted = m_curr*x+b_curr
    cost = 1/n* sum([val**2 for val in y-y_predicted ])
    md=-(2/n)*sum(x*(y-y_predicted))
    bd=-(2/n)*sum(y-y_predicted)
    m_curr = m_curr - learning_rate * md
    b_curr = b_curr - learning_rate * bd
    print(m_curr,b_curr,md,cost)

