import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import root

def f(x):
    return 1.0/(1.0+x**2);

def df(x):
    return -2.0*x*f(x)**2;

def h(x):
    return x*f(x);

def dh(x):
    return f(x)+x*df(x);

def saturate(x,sat=np.infty):
    if x > sat:
        return sat
    if x < -sat:
        return -sat
    return x;

class opinion:
    
    def __init__(self,N,eta=0.5):
        self.N = N
        
        self.sat = np.infty
        self.eta = eta 
        
        self.mu = 1.0
        self.kh = 0.0
        
        self.t1 = -np.infty
        
        self.atol = None # 1e-8
        self.rtol = None # 1e-8
        
        self.rootmethod = 'hybr'
        
    def control(self,X,t):
        
        phi = np.sum(X[self.N+1:])
        
        if t < self.t1:
            u = np.sign(phi)*self.sat
        else:
            u = phi/self.mu;
            
        return saturate(u,self.sat);


    def dynX(self,X,t):
        
        dX = np.zeros(X.shape)
        
        Y = X[1:self.N+1]
        P = X[self.N+1:]
        
        u = self.control(X,t)
        
        Yij = Y.reshape(-1,1)-Y.reshape(1,-1)
        Pij = P.reshape(-1,1)-P.reshape(1,-1)
        
        dX[0] = u
        
        dX[1:self.N+1] = self.kh*np.sum(h(Yij),axis=0) - h(X[1:self.N+1]) - u
        
        dX[self.N+1:] = P*dh(Y) + self.kh*np.sum(Pij*dh(Yij),axis=0)
        
        return dX;
    
    def odeint(self,X0,t):
        return odeint(self.dynX, X0, t, atol=self.atol, rtol=self.rtol)
    
    def hamil(self,X,t):
    
        dX = self.dynX(X,t)
        
        u = self.control(X,t)
        
        H = 1 + 0.5*self.mu*(u**2) + np.sum(X[self.N+1:]*dX[1:self.N+1]);
        
        return H;
    
    def F_zero(self,Z,Y0):
        P0 = Z[0:-1]
        tf = Z[-1]
        
        X0 = np.concatenate((Y0,P0))

        X = self.odeint(X0,(0.0,tf))
        Xf = X[-1,:]
        
        F = np.zeros(self.N+1)
        
        F[0] = self.hamil(Xf,tf)
        F[1] = Xf[1] + self.eta
        F[2] = Xf[self.N] - self.eta
        
#        if (self.N > 2):
#            F[3] = self.hamil(X0,0.0)

        return F;
    
    def solve(self,Y0,P0,tf0,trace=False,echo=False):
        
        Z0 = np.concatenate((P0,tf0))
        
        sol = root(self.F_zero,Z0,args=(Y0),method=self.rootmethod)
        
        Z1 = sol.x        
        sol.tf = np.array([Z1[-1]])
        sol.P = Z1[:-1]
        
        if echo:
            print(sol)
        
        if trace:
            self.trace(Y0,sol.P,sol.tf)        
        
        return sol, sol.P, sol.tf;
    
     # doesn't work because float are not mutable (no pass by reference)
#    def solve_c(self,Y0,P,tf,param,param_end,param_step,trace=False,echo=False):
#        
#        direction = np.sign(param_end-param)
#        
#        if direction > 0:
#            condition = param < param_end
#        else:
#            condition = param > param_end    
#        
#        while condition :
#    
#            param += direction*param_step
#            
#            if (direction > 0 and param > param_end) or (direction <= 0 and param < param_end):
#                param = param_end    
#                
#            print(param)
#            
#            sol, P, tf = self.solve(Y0,P,tf,trace,echo)
#            
#            print(sol.success)
#            
#            if direction > 0:
#                condition = param < param_end
#            else:
#                condition = param > param_end 
#            
#        return sol, P, tf;
    
    def trace(self,Y0,P0,tf):
        
        X0 = np.concatenate((Y0,P0))
        
        t = np.linspace(0.0,tf,200).reshape(-1,)
        X = self.odeint(X0,t)
        
        x0 = X[:,0].reshape(-1,1)
        xi = X[:,1:self.N+1]+x0
        
        u = np.zeros(t.shape)
        H = np.zeros(t.shape)
        
        for i in range(t.shape[0]):
            u[i] = self.control(X[i,:],t[i])
            H[i] = self.hamil(X[i,:],t[i])
        
        plt.figure(figsize=(12,15))
        plt.subplot(4,1,1)        
        plt.plot(t,x0,'r',label='leader')
        plt.plot(t,xi,'b',label='agents')
        plt.grid(True)
        plt.xlabel('time')
        plt.ylabel('opinions')
#        plt.legend()
        
        plt.subplot(4,1,2)
        plt.plot(t,u,'r')
        plt.grid(True)
        plt.xlabel('time')
        plt.ylabel('control')
        
        plt.subplot(4,1,3)
        plt.plot(t,X[:,self.N+1:],'c')
        plt.grid(True)  
        plt.xlabel('time')
        plt.ylabel('co-states')
        
        plt.subplot(4,1,4)
        plt.plot(t,H,'g')
        plt.grid(True)  
        plt.xlabel('time')
        plt.ylabel('hamiltonian')
        
        
        plt.show()

