import math as math
import numpy as np
import cvxopt as cv
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import minimize, Bounds
from scipy.linalg import orth
from dppy.finite_dpps import FiniteDPP #Requires the DPPy package by Guillaume Gautier
import sobol_seq as ss

marker='o'

def PHImono(dim,deg):
    if deg == 0:
        return lambda P: np.matrix(0*P[:,0]+1).T
    if dim == 1:
        return lambda P: np.power(P[:,0],np.asmatrix(np.arange(deg+1)).T).T
    def fun(P):
        Q = PHImono(dim-1,deg)(P[:,:-1]).T
        for j in range(deg):
            Q = np.vstack((Q,np.multiply(PHImono(dim-1,deg-j-1)(P[:,:-1]).T,P[:,-1]**(j+1))))
        return Q.T
    return fun

def PHImono_list(dim,deg):
    if deg==0:
        return np.zeros((1,dim))
    if dim==1:
        return np.array([np.arange(deg+1)]).T
    Q = PHImono_list(dim-1,deg)
    Q = np.hstack((Q,np.zeros((Q.shape[0],1))))
    for j in range(deg):
        R = PHImono_list(dim-1,deg-(j+1))
        R = np.hstack((R,j+1+np.zeros((R.shape[0],1))))
        Q = np.vstack((Q,R))
    return Q

def PHImono_show(dim,deg):
    if deg>=10:
        raise ValueError("Can't print superscripts higher than 9 because I haven't coded it yet.")
    if dim>=10:
        raise ValueError("Can't print subscripts higher than 9 because I haven't coded it yet.")
    Q = PHImono_list(dim,deg)
    print('1')
    if dim<=5:
        dico = ['x','y','z','t','u']
    else:
        dico = ['x\u2081','x\u2082','x\u2083','x\u2084','x\u2085','x\u2086','x\u2087','x\u2088','x\u2089']
    power = ['\u00b2','\u00b3','\u2074','\u2075','\u2076','\u2077','\u2078','\u2079']
    for i in range(1,Q.shape[0]):
        s = ''
        pol = Q[i,:]
        for j in range(pol.shape[0]):
            if pol[j]==1:
                s += dico[j]
            elif pol[j]>1:
                s += dico[j]+power[int(pol[j])-2]
        print(s)
        
def multinom_to_list(L):
    U = []
    for i in range(len(L)):
        if L[i]>0:
            U += [i for _ in range(int(L[i]))]
    return U
        
def DiscreteVS(PHI,P,nbpoint=None,crit="D"):
    
    Pt = np.unique([tuple(row) for row in P], axis=0)
    
    Q = PHI(Pt)
    V = cv.matrix(Q.T)
    d = V.size[0]
    n = V.size[1]
    G = cv.spmatrix(-1.0, range(n), range(n))
    h = cv.matrix(0.0, (n,1))
    b = cv.matrix(1.0)
    
    if nbpoint is None:
        nbpoint = d
    
    if crit=="D":
        A = cv.matrix(1.0, (1,n))
        def F(x=None, z=None):
            if x is None: return 0, cv.matrix(1.0, (n,1))
            X = V * cv.spdiag(x) * V.T
            L = +X
            try: cv.lapack.potrf(L)
            except ArithmeticError: return None
            f = - 2.0 * sum([math.log(L[i,i]) for i in range(d)])
            W = +V
            cv.blas.trsm(L, W)
            gradf = cv.matrix(-1.0, (1,d)) * W**2
            if z is None: return f, gradf
            H = cv.matrix(0.0, (n,n))
            cv.blas.syrk(W, H, trans='T')
            return f, gradf, z[0] * H**2
        xd = cv.solvers.cp(F, G, h, A = A, b = b)['x']        
        U = np.array(xd)[:,0]
    elif crit=="A":
        novars = n + int(d*(d+1)/2)
        c = cv.matrix(0.0, (novars,1))
        cst=n
        for a in range(d):
            c[cst]=1
            cst += d-a
        Ga = cv.matrix(0.0, (n, novars))
        Ga[:,:n] = G
        Gs = [cv.matrix(0.0, (4*d**2, novars))]
        for k in range(n):
            Gk = cv.matrix(0.0, (2*d,2*d))
            Gk[:d,:d] = -V[:,k] * V[:,k].T
            Gs[0][:,k] = Gk[:]
        cst=n
        for i in range(d):
            for j in range(i,d):
                Gk = cv.matrix(0.0, (2*d,2*d))
                Gk[d+i,d+j]=-1
                Gk[d+j,d+i]=-1
                Gs[0][:,cst] = Gk[:]
                cst += 1
        hs = [cv.matrix(0.0, (2*d,2*d))]
        hs[0][d:,:d] = cv.spmatrix(-1.0, range(d), range(d))
        Aa = cv.matrix(n*[1.0] + int(d*(d+1)/2)*[0.0], (1,novars))
        try:
            sol = cv.solvers.sdp(c, Ga, h, Gs, hs, Aa, b)
        except:
            return P[range(nbpoint),:]
        U = list(sol['x'][:n])
        
    U = np.abs(np.array(U))
    U /= sum(U)
    
    WeightedQ = ((np.diag(U)).dot(Q)).T
    OrthoQ = orth(WeightedQ.T)
    Ker = (OrthoQ).dot(OrthoQ.T)
    DPP = FiniteDPP(kernel_type='correlation', projection=True, K=Ker)
    T = DPP.sample_exact()   

    if nbpoint>d:
        T += multinom_to_list(np.random.multinomial(nbpoint-d,U))

    return Pt[T,:]

def Finite_ExM(OD,X, crit = "D" , progress = False):
    n = X.shape[0]
    ini = np.random.randint(0,n,size=OD.nbpoint)
    P = X[ini,:]
    k = 0
    while k<OD.nbpoint:
        Q = np.delete(P,0,0)
        old_val = OD.opt(P, crit = crit)
        L = [OD.opt(np.vstack((Q,X[i,:])), crit = crit) for i in range(n)]
        i = np.argmin(L)
        P = np.vstack((Q,X[i,:]))
        new_val = OD.opt(P, crit = crit)
        if abs(new_val-old_val) <= 0.00001: 
            k += 1
        else:
            k = 0
        if progress:
            print("\rNon-opt streak: "+str(k)+"/"+str(OD.nbpoint)+" | "+crit+"-Optimality of current best: "+str(new_val),end = "")
    return P

def RegP(xlim,ylim,npoint):
    dim = len(xlim)
    L = [np.linspace(xlim[i],ylim[i],npoint) for i in range(dim)]
    Mg = np.meshgrid(*L)
    Mgflat = [np.asmatrix(Mg[i].flatten()).T for i in range(dim)]
    Points = np.hstack(tuple(Mgflat))
    return np.asarray(Points)
    
class OptDesign:
    def __init__(self,PHI,lower_point,upper_point,nbpoint=None,cara=None,Urand=None,plot_fun=None,opt_1P_fun=None):
        
        l = np.array(lower_point)
        u = np.array(upper_point)
        if len(l)!=len(u):
            raise ValueError("Both extremum points should have the same dimension.")
        if not np.all(l<u):
            raise ValueError("lower_point < upper_point not satisfied.")
        self.lower_point = lower_point
        self.upper_point = upper_point  
        dim = len(lower_point)
        self.dim = dim
        
        self.PHI = PHI
            
        self.nbreg = PHI(np.zeros((1,dim))).shape[1]
        if nbpoint is None:
            self.nbpoint = self.nbreg
        elif nbpoint < self.nbreg:
            raise ValueError("The number of points of the design should be higher than the number of regressing functions")
        else: self.nbpoint = nbpoint
        
        if cara is None:
            def cara(P):
                Bool_Ar = np.logical_and(P[:,0] <= upper_point[0],P[:,0] >= lower_point[0])
                for i in range(1,dim):
                    Bool_Ar = np.logical_and(Bool_Ar,P[:,i] <= upper_point[i])
                    Bool_Ar = np.logical_and(Bool_Ar,P[:,i] >= lower_point[i])
                return(Bool_Ar)
        self.cara=cara
        
        if Urand is None and cara is None:
            def Urand(n):
                u=np.matrix(upper_point)
                l=np.matrix(lower_point)
                one=np.ones((n,dim))
                return np.array(np.multiply(one,l)+np.multiply(np.random.rand(n,dim),np.multiply(one,u-l)))
        
        if Urand is None and cara is not None:
            def Urand(n):
                u=np.matrix(upper_point)
                l=np.matrix(lower_point)
                one=np.ones((n,dim))
                P=np.array(np.multiply(one,l)+np.multiply(np.random.rand(n,dim),np.multiply(one,u-l)))
                P=P[cara(P),:]
                while P.shape[0]<n:
                    P2=np.array(np.multiply(one,l)+np.multiply(np.random.rand(n,dim),np.multiply(one,u-l)))  
                    P2=P2[cara(P2),:]
                    P=np.vstack((P,P2))
                return P[:n,:]
        self.Urand=Urand
        
        if plot_fun is None and dim==2:
            def plot_fun(ax):
                l=lower_point
                u=upper_point
                ax.plot([l[0],u[0],u[0],l[0],l[0]],[l[1],l[1],u[1],u[1],l[1]],'b')
                ax.axis("off")
                return ax
        self.plot_fun=plot_fun
        
        self.opt_1P_fun=opt_1P_fun
        
    def opt(self, x, crit="D"):
        P=x.reshape(self.nbpoint,self.dim)
        Q=self.PHI(P)
        if not all(self.cara(P)): 
            return(10000000000)
        if crit=="D": 
            return -np.linalg.slogdet((Q.T).dot(Q))[1]
        else:
            M=(Q.T).dot(Q)
            d=np.linalg.det(M)
            if d==0: 
                return(10000000000)
            size=M.shape[0]
            arr=np.arange(size)
            s=0
            for i in arr:
                N=M[arr!=i,:]
                s += np.linalg.det(N[:,arr!=i])
            return np.abs(s/d)
        
    def P_opt(self,P,x0,crit):
        if self.opt_1P_fun is not None:
            return self.opt_1P_fun(self.opt,P,x0,crit)
        else:
            bounds=Bounds(self.lower_point,self.upper_point)
            res = minimize(lambda x:self.opt(np.vstack((P,x)),crit=crit), x0, method='L-BFGS-B',bounds=bounds)
            return res.x
        
    def SS_generate(self,n):
        P=ss.i4_sobol_generate(self.dim, n)
        rand_shift=np.random.rand(1,self.dim)
        P=np.mod(P+rand_shift,1)
        u=np.matrix(self.upper_point)
        l=np.matrix(self.lower_point)
        one=np.ones((n,self.dim))
        P=np.array(np.multiply(one,l)+np.multiply(P,np.multiply(one,u-l)))
        return P
    
    def Gauss_generate(self,P,k,sigma,rwiter):
        DupP=np.vstack(tuple([P for i in range(k)]))
        n=DupP.shape[0]
        Pupd=np.copy(DupP)
        for _ in range(rwiter):
            #Pupd=DupP+np.random.normal(loc=0,scale=sigma,size=(n,self.dim))
            for i in range(n):
                while True:
                    Pupd[i,:]=DupP[i,:]+np.random.normal(loc=0,scale=sigma,size=(1,self.dim))
                    CARA=self.cara(np.reshape(Pupd[i,:],(1,self.dim)))
                    if CARA:
                        break
            DupP=np.copy(Pupd)
        return DupP
    
    def RandomTesting(self,n=1,crit="D"):
        for _ in range(n):
            print(self.opt(self.Urand(self.nbpoint)))
        
    def plot(self,Ptup,s=100,modify_plot=False):
        if type(Ptup)!=tuple:
            l=self.lower_point
            u=self.upper_point
            fig=plt.figure(figsize=(8,8*(u[1]-l[1])/(u[0]-l[0])))
            ax=fig.add_subplot(1,1,1)
            if self.dim!=2:
                raise ValueError("Plotting only available in dimension 2.")
            ax=self.plot_fun(ax)
            Pt, Nb = np.unique([tuple(row) for row in Ptup], axis=0,return_counts=True)
            ax.scatter(Pt[:,0],Pt[:,1],marker=marker,c='r',s=s*Nb)
        else:
            n=len(Ptup)
            l=self.lower_point
            u=self.upper_point
            fig=plt.figure(figsize=(8*n,8*(u[1]-l[1])/(u[0]-l[0])))
            for i in range(n):
                ax=fig.add_subplot(1,n,i+1)
                ax=self.plot_fun(ax)
                Pt, Nb = np.unique([tuple(row) for row in Ptup[i]], axis=0,return_counts=True)
                ax.scatter(Pt[:,0],Pt[:,1],marker=marker,c='r',s=s*Nb)
        if modify_plot:
            plt.close()
            return fig
        else:
            plt.show()
                
    
#############################Classe ARIFAA##############################
        
class DOGS:
    def __init__(self,OD,nbupd = None,ini = None,crit = "D", CP = None):
        
        if crit not in ["D","A"]:
            raise ValueError("Wrong optimality criterion. It should be A or D.")
                            
        if CP is None:
            CP = np.array([[]])
            CP.shape = (0,OD.dim)
        self.CP = CP
        
        self.OD = OD
        self.crit = crit
        self.ini = ini
        self.nbupd = OD.nbreg if nbupd is None else nbupd
        
    def simulate(self,nbiter,progress=False):
        
        cv.solvers.options['show_progress']=False

        if self.ini is None:
            P=self.OD.Urand(self.OD.nbpoint)
        else:
            P=self.ini
        
        for N in range(nbiter):
            oldval = self.OD.opt(P,crit = self.crit)
            Pupd = self.OD.Urand(self.nbupd)
            try:
                Pfin = DiscreteVS(self.OD.PHI,np.vstack((P,Pupd,self.CP)),nbpoint = self.OD.nbpoint,crit = self.crit)
            except:
                Pfin = P
            newval = self.OD.opt(Pfin,crit = self.crit)
            if newval<oldval : 
                P = np.copy(Pfin)
            if progress: 
                print('\rProgress: '+f'{100*(N+1)/nbiter:.2f}'+'%', end = " ")
        if progress: 
            print('Done')
        return P
    
    def simulate_extreme(self,nbiter,progress=False,Fexm_progress=False):
        P = self.OD.Urand(self.OD.nbpoint)
        
        for N in range(nbiter):
            oldval = self.OD.opt(P,crit=self.crit)
            Pupd = self.OD.Urand(self.nbupd)
            X = np.unique([np.vstack((P,Pupd,self.CP)) for row in P], axis=0)[0]
            Pfin = Finite_ExM(self.OD,X,progress = Fexm_progress,crit = self.crit)
            newval = self.OD.opt(Pfin,crit=self.crit)
            if newval<oldval: 
                P = np.copy(Pfin)
            if progress: 
                print('\rProgress: '+f'{100*(N+1)/nbiter:.2f}'+'%', end =" ")
        if progress: 
            print('Done')
        
        return(P)
            
    def testing(self,nbiter,nbtest=1,progress=False):
                
        cv.solvers.options['show_progress']=False
        
        All_result=np.zeros(nbiter+1)
        
        for N in range(nbtest):
        
            if self.ini is None:
                P=self.OD.Urand(self.OD.nbpoint)
            else:
                P=self.ini

            L=[self.OD.opt(P,crit=self.crit)]
    
            for _ in range(nbiter):
                oldval=self.OD.opt(P,crit=self.crit)
                Pupd=self.OD.Urand(self.nbupd)
                try:
                    Pfin=DiscreteVS(self.OD.PHI,np.vstack((P,Pupd,self.CP)),nbpoint=self.OD.nbpoint,crit=self.crit)
                except:
                    Pfin=P
                newval=self.OD.opt(Pfin,crit=self.crit)
                if newval<oldval : 
                    P=np.copy(Pfin)
                L=L+[self.OD.opt(P,crit=self.crit)]
                
            All_result=np.vstack((All_result,np.asarray(L)))
            if progress: 
                print('\rProgress: '+f'{100*(N+1)/nbtest:.2f}'+'%', end =" ")
        if progress: 
            print('Done')
        return(All_result[1:,:])
        
    def animate(self,nbiter,nbframe_ini=10,fps=10,show_last_frame=False):
        
        if self.OD.dim!=2:
            raise ValueError('Animation only available in dimension 2')
        
        l=self.OD.lower_point
        u=self.OD.upper_point
        fig=plt.figure(figsize=(10,10*(u[1]-l[1])/(u[0]-l[0])))
        ax=fig.add_subplot(1,1,1)
        ax=self.OD.plot_fun(ax)
        line2 = ax.scatter([], [], marker=marker, s=100, alpha=0.2, color="grey")
        line = ax.scatter([], [], marker=marker, s=100, color='r')
        leg1 = ax.text(self.OD.lower_point[0],self.OD.upper_point[1]+(self.OD.upper_point[1]-self.OD.lower_point[1])/100,"", fontsize=20)
        leg2 = ax.text(self.OD.lower_point[0],self.OD.upper_point[1]+(self.OD.upper_point[1]-self.OD.lower_point[1])/10,"", fontsize=20)
        
        def init():
            global P
            if self.ini is None:
                P=self.OD.Urand(self.OD.nbpoint)
            else:
                P=self.ini
            cv.solvers.options['show_progress']=False
            line2.set_offsets(np.array([[],[]]).T)
            line.set_offsets(np.array([[],[]]).T)
            leg1.set_text("")
            leg2.set_text("")
            return line,
        
        def anim_fun(Niter):
            global P
            Nframe = np.copy(Niter)
            if Nframe<nbframe_ini:
                line.set_offsets(P)
                leg1.set_text("Iteration 0")
                leg2.set_text("log(h_D(X)) = "+ "%.3f" % self.OD.opt(P,crit=self.crit))
                return line2, line,
            
            oldval=self.OD.opt(P,crit=self.crit)
            Pupd=self.OD.Urand(self.nbupd)
            try:
                Pfin=DiscreteVS(self.OD.PHI,np.vstack((P,Pupd,self.CP)),nbpoint=self.OD.nbpoint,crit=self.crit)
            except:
                Pfin=P
            newval=self.OD.opt(Pfin,crit=self.crit)
            if newval<oldval : 
                P=np.copy(Pupd)          
            Pt, Nb = np.unique([tuple(row) for row in P], axis=0,return_counts=True)            
            line.set_offsets(Pt)
            line.set_sizes(100*Nb)
            line2.set_offsets(Pupd)
            leg1.set_text("Iteration "+str(Nframe-nbframe_ini+1))
            leg2.set_text("log(h_D(X)) = "+ "%.3f" % self.OD.opt(P,crit=self.crit))
            return line2, line,

        anim = animation.FuncAnimation(fig, anim_fun, init_func=init, frames=nbiter+nbframe_ini, blit=True)
        anim.save('ARIFAA_animation.mp4', fps=fps)
        if not show_last_frame: plt.close()
        
#############################Classe LSA##############################
        
class LSA:
    def __init__(self,OD,sd,ini=None,crit="D"):
            
        self.OD=OD
        self.sd=sd
        self.ini=ini
        self.crit=crit
        
    def simulate(self,nbiter,progress=False):
        
        if self.ini is None:
            P=self.OD.Urand(self.OD.nbpoint)
        else: P=self.ini
            
        for N in range(nbiter):
            oldval=self.OD.opt(P,crit=self.crit)
            Pupd=P+np.random.normal(loc=0,scale=self.sd,size=(self.OD.nbpoint,self.OD.dim))
            CARA=self.OD.cara(Pupd)
            for i in range(self.OD.nbpoint):
                if not CARA[i]:
                    Pupd[i,:]=np.copy(P[i,:])
            newval=self.OD.opt(Pupd,crit=self.crit)
            if newval<oldval : 
                P=np.copy(Pupd)
            if progress: 
                print('\rProgress: '+f'{100*(N+1)/nbiter:.2f}'+'%', end =" ")
        if progress: 
            print('Done')
        return P
        
    def testing(self,nbiter,nbtest=1,progress=False):
        
        All_result=np.zeros(nbiter+1)
        
        for N in range(nbtest):
        
            if self.ini is None:
                P=self.OD.Urand(self.OD.nbpoint)
            else:
                P=self.ini

            L=[self.OD.opt(P,crit=self.crit)]
    
            for _ in range(nbiter):
                oldval=self.OD.opt(P,crit=self.crit)
                Pupd=P+np.random.normal(loc=0,scale=self.sd,size=(self.OD.nbpoint,self.OD.dim))
                CARA=self.OD.cara(Pupd)
                for i in range(self.OD.nbpoint):
                    if not CARA[i]:
                        Pupd[i,:]=np.copy(P[i,:])
                newval=self.OD.opt(Pupd,crit=self.crit)
                if newval<oldval : 
                    P=np.copy(Pupd)
                L=L+[self.OD.opt(P,crit=self.crit)]
                
            All_result=np.vstack((All_result,np.asarray(L)))
            if progress: print('\rProgress: '+f'{100*(N+1)/nbtest:.2f}'+'%', end =" ")
        if progress: print('Done')
        return(All_result[1:,:])
        
    def animate(self,nbiter,nbframe_ini=20,fps=10,show_last_frame=False):
        
        if self.OD.dim!=2:
            raise ValueError('Animation only available in dimension 2')
        
        l=self.OD.lower_point
        u=self.OD.upper_point
        fig=plt.figure(figsize=(10,10*(u[1]-l[1])/(u[0]-l[0])))
        ax=fig.add_subplot(1,1,1)
        ax=self.OD.plot_fun(ax)
        line2 = ax.scatter([], [], marker=marker, s=100, alpha=0.2, color="grey")
        line = ax.scatter([], [], marker=marker, s=100, color='r')
        leg1 = ax.text(self.OD.lower_point[0],self.OD.upper_point[1]+(self.OD.upper_point[1]-self.OD.lower_point[1])/50,"", fontsize=20)
        leg2 = ax.text(self.OD.lower_point[0],self.OD.upper_point[1]+(self.OD.upper_point[1]-self.OD.lower_point[1])/10,"", fontsize=20)
        
        def init():
            global P
            if self.ini is None:
                P=self.OD.Urand(self.OD.nbpoint)
            else:
                P=self.ini
            line2.set_offsets(np.array([[],[]]).T)
            line.set_offsets(np.array([[],[]]).T)
            leg1.set_text("")
            leg2.set_text("")
            return line,
        
        def anim_fun(Niter):
            global P
            Nframe = np.copy(Niter)
            if Nframe<nbframe_ini:
                line.set_offsets(P)
                line2.set_offsets(np.array([[],[]]).T)
                leg1.set_text("Iteration 0")
                leg2.set_text("log(h_D(X)) = "+ "%.3f" % self.OD.opt(P,crit=self.crit))
                return line2, line,
            
            oldval=self.OD.opt(P,crit=self.crit)
            Pupd=P+np.random.normal(loc=0,scale=self.sd,size=(self.OD.nbpoint,self.OD.dim))
            CARA=self.OD.cara(Pupd)
            for i in range(self.OD.nbpoint):
                if not CARA[i]:
                    Pupd[i,:]=np.copy(P[i,:])
            newval=self.OD.opt(Pupd,crit=self.crit)
            if newval<oldval : 
                P=np.copy(Pupd)
            Pt, Nb = np.unique([tuple(row) for row in P], axis=0,return_counts=True)            
            line.set_offsets(Pt)
            line.set_sizes(100*Nb)
            line2.set_offsets(Pupd)
            leg1.set_text("Iteration "+str(Nframe-nbframe_ini+1))
            leg2.set_text("log(h_D(X)) = "+ "%.3f" % self.OD.opt(P,crit=self.crit))
            return line2, line,

        anim = animation.FuncAnimation(fig, anim_fun, init_func=init, frames=nbiter+nbframe_ini, blit=True)
        anim.save('LSA_animation.mp4', fps=fps)
        if not show_last_frame: plt.close()
    
#############################Classe ExM##############################
        
class ExM:
    def __init__(self,OD,ini=None,crit="D"):
        
        if crit not in ["D","A"]:
            raise ValueError("Wrong optimality criterion. It should be A or D.")
        
        self.OD=OD
        self.ini=ini
        self.crit=crit
        
    def simulate(self,nbiter,progress=False):

        if self.ini is None:
            P=self.OD.Urand(self.OD.nbpoint)
        else: P=self.ini
            
        for N in range(nbiter):
            x0=P[0,:]
            Q=np.delete(P,0,0)
            x = self.OD.P_opt(Q,x0,self.crit)
            P=np.vstack((Q,x))
            if progress: print('\rProgress: '+f'{100*(N+1)/nbiter:.2f}'+'%', end =" ")
        if progress: print('Done')
        return(P)
        
    def testing(self,nbiter,nbtest=1,progress=False):
        
        All_result=np.zeros(nbiter+1)
        
        for N in range(nbtest):
        
            if self.ini is None:
                P=self.OD.Urand(self.OD.nbpoint)
            else:
                P=self.ini

            L=[self.OD.opt(P,crit=self.crit)]
    
            for _ in range(nbiter):
                x0=P[0,:]
                Q=np.delete(P,0,0)
                x=self.OD.P_opt(Q,x0,self.crit)
                P=np.vstack((Q,x))
                L=L+[self.OD.opt(P,crit=self.crit)]
                
            All_result=np.vstack((All_result,np.asarray(L)))
            if progress: 
                print('\rProgress: '+f'{100*(N+1)/nbtest:.2f}'+'%', end =" ")
        if progress: 
            print('Done')
        return All_result[1:,:]
        
    def animate(self,nbiter,nbframe_ini=20,fps=10,show_last_frame=False):
        
        if self.OD.dim!=2:
            raise ValueError('Animation only available in dimension 2')
        
        l=self.OD.lower_point
        u=self.OD.upper_point
        fig=plt.figure(figsize=(10,10*(u[1]-l[1])/(u[0]-l[0])))
        ax=fig.add_subplot(1,1,1)
        ax=self.OD.plot_fun(ax)
        line2 = ax.scatter([], [], marker=marker, s=100, alpha=0.2, color="grey")
        line = ax.scatter([], [], marker=marker, s=100, color='r')
        leg1 = ax.text(self.OD.lower_point[0],self.OD.upper_point[1]+(self.OD.upper_point[1]-self.OD.lower_point[1])/100,"", fontsize=20)
        leg2 = ax.text(self.OD.lower_point[0],self.OD.upper_point[1]+(self.OD.upper_point[1]-self.OD.lower_point[1])/10,"", fontsize=20)
        
        def init():
            global P
            if self.ini is None:
                P=self.OD.Urand(self.OD.nbpoint)
            else:
                P=self.ini
            line2.set_offsets(np.array([[],[]]).T)
            line.set_offsets(np.array([[],[]]).T)
            leg1.set_text("")
            leg2.set_text("")
            return line,
        
        def anim_fun(Niter):
            global P
            Nframe = np.copy(Niter)
            if Nframe<nbframe_ini:
                line.set_offsets(P)
                line2.set_offsets(np.array([[],[]]).T)
                leg1.set_text("Iteration 0")
                leg2.set_text("log(h_D(X)) = "+ "%.3f" % self.OD.opt(P,crit=self.crit))
                return line2, line,
            
            x0=P[0,:]
            Q=np.delete(P,0,0)
            x=self.OD.P_opt(Q,x0,self.crit)
            P=np.vstack((Q,x))
            Pt, Nb = np.unique([tuple(row) for row in P], axis=0,return_counts=True)            
            line.set_offsets(Pt)
            line.set_sizes(100*Nb)
            line2.set_offsets(np.array([[],[]]).T)
            leg1.set_text("Iteration "+str(Nframe-nbframe_ini+1))
            leg2.set_text("log(H_D(X)) = "+ "%.3f" % self.OD.opt(P,crit=self.crit))
            return line2, line,

        anim = animation.FuncAnimation(fig, anim_fun, init_func=init, frames=nbiter+nbframe_ini, blit=True)
        anim.save('ExM_animation.mp4', fps=fps)
        if not show_last_frame: plt.close()
        
#############################Classe DisSpace##############################
class Discrete_ExM:
    def __init__(self,OD,size=0,crit="D",CP=None):
        
        if crit not in ["D","A"]:
            raise ValueError("Wrong optimality criterion. It should be A or D.")
            
        if size==0 and CP is None:
          raise ValueError("If size=0 then CP must be specified")  
        
        self.OD=OD
        self.size=size
        self.crit=crit
        self.CP=CP
        
    def show(self):
        
        if self.OD.dim!=2:
            raise ValueError("Plotting only available in dimension 2.")
            
        P=RegP(self.OD.lower_point,self.OD.upper_point,self.size)
        if self.CP is not None:
            P=np.vstack((P,self.CP)) 
        P=P[self.OD.cara(P),:]
        self.OD.plot(P)
        
    def num_point(self):
            
        P=RegP(self.OD.lower_point,self.OD.upper_point,self.size)
        if self.CP is not None:
            P=np.vstack((P,self.CP))
        return(sum(self.OD.cara(P)))
        
    def simulate(self,progress=False):
        
        P=RegP(self.OD.lower_point,self.OD.upper_point,self.size)
        if self.CP is not None:
            P=np.vstack((P,self.CP)) 
        P=P[self.OD.cara(P),:] 
            
        return Finite_ExM(self.OD,P,progress = progress, crit = self.crit)
    
#############################Making old stuff still work##############################

def ARIFAA(OD,nbupd = None,ini = None,crit = "D",CP = None):
    return DOGS(OD,nbupd = nbupd,ini = ini,crit = crit,CP = CP)

def DisSpace(OD,size = 0,crit = "D",CP = None):
    return Discrete_ExM(OD,size = size,crit = crit,CP = CP)