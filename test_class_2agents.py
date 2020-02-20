import numpy as np
import opinion_class as op

#Y0 = np.array([0.0,4.0,12.0])
Y0 = np.array([0.0,3.0,14.0])

my_op = op.opinion(Y0)

my_op.kh = 0.0
my_op.mu = 1.0
my_op.sat = np.infty
my_op.eta = 0.5
my_op.t1 = 0.0

#my_op.atol = 1e-14
#my_op.rtol = 1e-10

P0 = np.zeros((my_op.N,))
P0[0] = -0.491088
P0[-1] = 1.91331

tf0 = np.array([16.67589717602594845])

print("Initial educated guess")
my_op.trace(P0,tf0)

print("Refining around guess")
sol0, P1, tf1 = my_op.solve(P0,tf0,trace=True,echo=False)

#%%
''' interactions '''
P2 = P1.copy()
tf2 = tf1.copy()

my_op.kh = 0.0
my_op.mu = 1.0
my_op.sat = np.infty

kh_end = 1.0
kh_step = 0.1

while my_op.kh < kh_end:
    
    my_op.kh += kh_step
    if my_op.kh > kh_end:
        my_op.kh = kh_end
    print(my_op.kh)
    
    sol1, P2, tf2 = my_op.solve(P2,tf2,trace=False,echo=False)
    print(sol1.success)

print("Solution with interactions")
my_op.trace(P2,tf2)

#%%
''' control '''
P3 = P2.copy()
tf3 = tf2.copy()

my_op.sat = 2.0
my_op.mu = 1.0
mu_end = 0.2
mu_step = 0.05

while my_op.mu > mu_end:
    
    my_op.mu -= mu_step
    if my_op.mu < mu_end:
        my_op.mu = mu_end 
    print(my_op.mu)
    
    sol2, P3, tf3 = my_op.solve(P3,tf3,trace=False,echo=False)
    print(sol2.success)

print("Solution without control cost")
my_op.trace(P3,tf3)

#%%
P4 = P3.copy()
tf4 = tf3.copy()

my_op.t1 = 0.0
my_op.mu = 0.2
mu_end = 0.125
mu_step = 0.005

while my_op.mu > mu_end:
    
    my_op.mu -= mu_step
    if my_op.mu < mu_end:
        my_op.mu = mu_end 
    print(my_op.mu)
    
    sol3, P4, tf4 = my_op.solve(P4,tf4,trace=False,echo=False)
    print(sol3.success)

#%%
my_op.t1 = 0.0
print("Solution without control cost")
data4 = my_op.trace(P4,tf4)
tk4 = data4.t
u4 = data4.u
Pk4 = data4.P


#%%
''' switching time '''

my_op.mu = 0.125

# educated guess
id_t1 = 21
my_op.t1 = tk4[id_t1]
P5 = Pk4[id_t1,:]

Z4 = np.zeros((my_op.N+2,))
Z4[:my_op.N] = P5
Z4[-2] = my_op.t1
Z4[-1] = tf4

#F = my_op.F_zero_sw(Z4)

#my_op.trace_sw(P5,tf4)

my_op.rootmethod = 'lm'
sol4, P5, tf4 = my_op.solve_sw(P5,tf4,trace=False,echo=False)
print(sol4.success)

my_op.trace_sw(P5,tf4)

my_op.mu = 0.125
mu_end = 0.1
mu_step = 0.001

while my_op.mu > mu_end:
    
    my_op.mu -= mu_step
    if my_op.mu < mu_end:
        my_op.mu = mu_end 
    print(my_op.mu)
    
    sol4, P5, tf4 = my_op.solve_sw(P5,tf4,trace=True,echo=True)
    print(sol4.success)

my_op.trace_sw(P5,tf4)