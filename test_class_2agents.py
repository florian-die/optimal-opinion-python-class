import numpy as np
import opinion_class as op

N = 2

my_op = op.opinion(N)

my_op.kh = 0.0
my_op.mu = 1.0
my_op.sat = np.infty
my_op.eta = 0.5

#my_op.atol = 1e-14
#my_op.rtol = 1e-10

#Y0 = np.array([0.0,4.0,12.0])
Y0 = np.array([0.0,3.0,14.0])
P0 = np.zeros((N,))
P0[0] = -0.491088
P0[-1] = 1.91331

tf0 = np.array([16.67589717602594845])

X0 = np.concatenate((Y0,P0))
Z0 = np.concatenate((P0,tf0))

print("Initial educated guess")
my_op.trace(Y0,P0,tf0)

print("Refining around guess")
sol0, P1, tf1 = my_op.solve(Y0,P0,tf0,trace=True,echo=False)

#%%
''' interactions '''
P2 = P1.copy()
tf2 = tf1.copy()

my_op.kh = 0.0
my_op.mu = 1.0
my_op.sat = np.infty

kh_end = 1.0
kh_step = 0.1

#sol1, P2, tf2 = my_op.solve_c(Y0,P2,tf2,my_op.kh,kh_end,kh_step,trace=False,echo=False)
while my_op.kh < kh_end:
    
    my_op.kh += kh_step
    if my_op.kh > kh_end:
        my_op.kh = kh_end
    print(my_op.kh)
    
    sol1, P2, tf2 = my_op.solve(Y0,P2,tf2,trace=False,echo=False)
    print(sol1.success)

print("Solution with interactions")
my_op.trace(Y0,P2,tf2)

#%%
''' control '''
P3 = P2.copy()
tf3 = tf2.copy()

my_op.sat = 2.0
my_op.mu = 1.0
mu_end = 0.2
mu_step = 0.05

#sol2, P3, tf3 = my_op.solve_c(Y0,P2,tf2,my_op.mu,mu_end,mu_step,trace=False,echo=False)

while my_op.mu > mu_end:
    
    my_op.mu -= mu_step
    if my_op.mu < mu_end:
        my_op.mu = mu_end 
    print(my_op.mu)
    
    sol2, P3, tf3 = my_op.solve(Y0,P3,tf3,trace=False,echo=False)
    print(sol2.success)

print("Solution without control cost")
my_op.trace(Y0,P3,tf3)

#%%
P4 = P3.copy()
tf4 = tf3.copy()

my_op.mu = 0.2
mu_end = 0.125
mu_step = 0.005

#sol2, P3, tf3 = my_op.solve_c(Y0,P2,tf2,my_op.mu,mu_end,mu_step,trace=False,echo=False)

while my_op.mu > mu_end:
    
    my_op.mu -= mu_step
    if my_op.mu < mu_end:
        my_op.mu = mu_end 
    print(my_op.mu)
    
    sol3, P4, tf4 = my_op.solve(Y0,P4,tf4,trace=False,echo=False)
    print(sol3.success)

print("Solution without control cost")
my_op.trace(Y0,P4,tf4)