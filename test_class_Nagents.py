import numpy as np
import opinion_class as op

N = 6

my_op = op.opinion(N)

my_op.kh = 0.0
my_op.mu = 1.0
my_op.sat = np.infty
my_op.eta = 0.5

my_op.rootmethod = 'lm'

#my_op.atol = 1e-14
#my_op.rtol = 1e-10

Y0 = np.array([0.0,3,3.5,4.0,12.0,13.0,14.0])
P0 = np.zeros((N,))
P0[0] = -0.491088
P0[-1] = 1.91331

tf0 = np.array([16.67589717602594845])

X0 = np.concatenate((Y0,P0))
Z0 = np.concatenate((P0,tf0))

print("Initial educated guess")
my_op.trace(Y0,P0,tf0)

print("Refining around guess")
sol0, P1, tf1 = my_op.solve(Y0,P0,tf0,trace=True,echo=True)

#%%
''' interactions '''
P2 = P1.copy()
P2[1:-1] = 0.01
#P2[2] = 0.01
#P2[3] = 0.02
#P2[4] = 0.03
#P2[5] = 0.04
tf2 = tf1.copy()

my_op.mu = 1.0
my_op.kh = 0.0
my_op.sat = np.infty

my_op.rootmethod = 'lm'
#my_op.rootmethod = 'hybr'

kh_end = 1.0
kh_step = 0.1

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

#my_op.rootmethod = 'hybr'
my_op.rootmethod = 'lm'

my_op.sat = 2.0
my_op.mu = 1.0
my_op.kh = 1.0

mu_end = 0.1
mu_step = 0.1

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

mu_end = 0.01
mu_step = 0.01

while my_op.mu > mu_end:
    
    my_op.mu -= mu_step
    if my_op.mu < mu_end:
        my_op.mu = mu_end    
    print(my_op.mu)
    
    sol3, P4, tf4 = my_op.solve(Y0,P4,tf4,trace=False,echo=False)
    print(sol2.success)

print("Solution without control cost")
my_op.trace(Y0,P4,tf4)