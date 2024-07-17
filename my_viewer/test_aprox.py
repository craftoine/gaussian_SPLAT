import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import erfinv
#import a librairie for good integral aproximation
#use the module : scipy.integrate.quad
from scipy.integrate import quad

negligeable_val_when_exp = -6
min_sum_eval = 0.0001
rescaling = 0
#gaussian function
class Gaussian:
    def __init__(self, mu, sigma,weight,color):
        self.mu = mu
        self.sigma = sigma
        self.weight = weight
        self.color = color
    def __call__(self, x):
        return self.weight*np.exp(-0.5*((x-self.mu)/self.sigma)**2)/np.sqrt(2*np.pi*self.sigma**2)
    def integral(self,a,b):
        v = 1
        if rescaling == 1:
            if a == self.get_start():
                a = -np.inf
            if b == self.get_end():
                b = np.inf
        elif rescaling == 2:
            v = (self.weight)/(self.weight*((math.erf((self.get_end()-self.mu)/(np.sqrt(2)*self.sigma))-math.erf((self.get_start()-self.mu)/(np.sqrt(2)*self.sigma)))/2))
        #use math.erf for integral of gaussian
        return v*self.weight*((math.erf((b-self.mu)/(np.sqrt(2)*self.sigma))-math.erf((a-self.mu)/(np.sqrt(2)*self.sigma)))/2)
    def get_start(self):
        """v = -2 * (negligeable_val_when_exp - np.log(self.weight) - np.log(np.sqrt(2*np.pi*self.sigma**2)))*self.sigma*self.sigma
        if v>0:
            return self.mu - np.sqrt(v)
        else:
            return self.mu"""
        c = 1/(2*self.sigma*self.sigma)
        sqr_c = 1/(np.sqrt(2)*self.sigma)
        v = 1-(np.exp(negligeable_val_when_exp-np.log(self.weight))*sqr_c/(np.sqrt(np.pi)))
        if v>0:
            #print(v, erfinv(v))
            return self.mu - erfinv(v)/sqr_c
        else:
            return self.mu
    def get_end(self):
        """v = -2 * (negligeable_val_when_exp - np.log(self.weight) - np.log(np.sqrt(2*np.pi*self.sigma**2)))*self.sigma*self.sigma
        if v>0:
            return self.mu + np.sqrt(v)
        else:
            return self.mu"""
        c = 1/(2*self.sigma*self.sigma)
        sqr_c = 1/(np.sqrt(2)*self.sigma)
        v = 1-(np.exp(negligeable_val_when_exp-np.log(self.weight))*sqr_c/(np.sqrt(np.pi)))
        if v>0:
            return self.mu + erfinv(v)/sqr_c
        else:
            return self.mu
        
        
class GaussianMixture:
    def __init__(self, gaussians):
        self.gaussians = gaussians
    def __call__(self, x):
        return sum(g(x)*g.color for g in self.gaussians)
    def integral(self,a,b):
        return sum(g.integral(a,b) for g in self.gaussians)
    def get_fct(self,c):
        #f = lambda x: self(x)*np.exp(self.integral(0,x))*sum(g(x)*g.color.c for g in self.gaussians)/self(x)
        f = lambda x: np.exp(-self.integral(0,x))*self(x)[c]
        return f
    def call2(self,x):
        return sum(g(x) for g in self.gaussians)
#set random seed
random.seed(42)
nb_gaussians = 7
gaussians = [Gaussian(random.uniform(0,8), random.uniform(0.1,1),random.uniform(0.1,2),np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])) for _ in range(nb_gaussians)]
gm = GaussianMixture(gaussians)
f_0 = gm.get_fct(0)
f_1 = gm.get_fct(1)
f_2 = gm.get_fct(2)
start = 0
end = 10
#plot the gaussian mixture and the 3 functions
x = np.linspace(start,end,1000)
plt.plot(x,[gm(x_)[0] for x_ in x],label='red mixture', color = 'red')
plt.plot(x,[gm(x_)[1] for x_ in x],label='green mixture', color = 'green')
plt.plot(x,[gm(x_)[2] for x_ in x],label='blue mixture', color = 'blue')
plt.plot(x,[gm.call2(x_) for x_ in x],label='opacity mixture', color = 'black')
plt.xlabel('x')
plt.title('Gaussian Mixture value')
plt.legend()
plt.show()
plt.plot(x,[f_0(x_) for x_ in x],label='transmitance multiplied by red mixture', color = 'red')
plt.plot(x,[f_1(x_) for x_ in x],label='transmitance multiplied by green mixture', color = 'green')
plt.plot(x,[f_2(x_) for x_ in x],label='transmitance multiplied by blue mixture', color = 'blue')
plt.xlabel('x')
plt.title('values to integrate')
plt.legend()
plt.show()


k = 10
K = k
#compute the integral of f1,f2,f3 and plot them value
x = np.linspace(start,end,300)
plt.plot(x,[quad(f_0,start,x_)[0] for x_ in x],label='red true', color = 'lightcoral',alpha=0.7,linestyle='solid',linewidth=4)
plt.plot(x,[quad(f_1,start,x_)[0] for x_ in x],label='green true', color = 'lightgreen',alpha=0.7,linestyle='solid',linewidth=4)
plt.plot(x,[quad(f_2,start,x_)[0] for x_ in x],label='blue true', color = 'lightblue',alpha=0.7,linestyle='solid',linewidth=4)

def bad_integration_method(gm,start,end):
    l = []
    for i in range(len(gm.gaussians)):
        l.append((gm.gaussians[i].mu,i))
    l.sort(key = lambda x: x[0])
    x =[]
    y = []
    x.append(start)
    y.append(np.array([0.,0.,0.]))
    y.append(np.array([0.,0.,0.]))
    T = 1
    for i in range(len(l)):
        v,j = l[i]
        if start<=v<=end:
            eval = gm.gaussians[j].integral(start,end)
            alpha = 1-np.exp(-eval)
            r = y[len(y)-1] + (alpha*T)*gm.gaussians[j].color
            T = T * (1-alpha)
            x.append(v-0.000001)
            x.append(v)
            y.append(r)
            y.append(r)

    x.append(end)
    return x,y
def bad_integration_method_(gm,start,end):
    l = []
    for i in range(len(gm.gaussians)):
        l.append((gm.gaussians[i].mu,i))
    l.sort(key = lambda x: x[0])
    T = 1
    r = np.array([0.,0.,0.])
    for i in range(len(gm.gaussians)):
        v,j = l[i]
        eval = gm.gaussians[j].integral(start,end)
        alpha = 1-np.exp(-eval)
        r = r + (alpha*T)*gm.gaussians[j].color
        T = T * (1-alpha)

    return r
def my_integration_method(gm,start,end,k):
    l = []
    for i in range(len(gm.gaussians)):
        g = gm.gaussians[i]
        a = g.get_start()
        b = g.get_end()
        l.append((a,i,0))
        l.append((b,i,1))
    l.sort(key = lambda x: x[0] + 0.000001*x[2])
    x =[]
    y = []
    x.append(start)
    y.append(np.array([0.,0.,0.]))
    a = 0
    s = set()
    T=1
    #print(l)
    for i in range(len(l)):
        v,j,k_ = l[i]
        b = max(min(v,end),a)
        if a<b:
            step = (b-a)/k
            for m in range(k):
                sum_eval = 0
                color_add = np.array([0.,0.,0.])
                for n in s:
                    g = gm.gaussians[n]
                    eval  = g.integral(a+m*step,a+(m+1)*step)
                    sum_eval += eval
                    color_add += g.color*eval
                if sum_eval> min_sum_eval:
                    alpha = 1-np.exp(-sum_eval)
                    r = y[len(y)-1] + (alpha*T/sum_eval)*color_add
                    T = T * (1-alpha)
                    x.append(a+m*step)
                    y.append(r)
        a = b
        if k_==1:
            #print("remove",j)
            s.remove(j)
        else:
            #print("add",j)
            s.add(j)
    x.append(end)
    y.append(y[len(y)-1])
    return x,y
def my_integration_method_(gm,start,end,k):
    l = []
    for i in range(len(gm.gaussians)):
        g = gm.gaussians[i]
        a = g.get_start()
        b = g.get_end()
        l.append((a,i,0))
        l.append((b,i,1))
    l.sort(key = lambda x: x[0] + 0.000001*x[2])
    a = 0
    s = set()
    T=1
    r = np.array([0.,0.,0.])
    for i in range(len(l)):
        v,j,k_ = l[i]
        b = max(min(v,end),a)
        if a<b:
            step = (b-a)/k
            for m in range(k):
                sum_eval = 0
                color_add = np.array([0.,0.,0.])
                for n in s:
                    g = gm.gaussians[n]
                    eval  = g.integral(a+m*step,a+(m+1)*step)
                    sum_eval += eval
                    color_add += g.color*eval
                if sum_eval> min_sum_eval:
                    alpha = 1-np.exp(-sum_eval)
                    r = r + (alpha*T/sum_eval)*color_add
                    T = T * (1-alpha)
        a = b
        if k_==1:
            s.remove(j)
        else:
            s.add(j)
    return r
#x,y = my_integration_method(gm,start,end,k)
_f = lambda x: my_integration_method_(gm,start,x,k)

y = list(map(_f,x))

plt.plot(x,[y_[0] for y_ in y],label='red overlapping', color = 'red',linestyle='dotted')
plt.plot(x,[y_[1] for y_ in y],label='green overlapping', color = 'green',linestyle='dotted')
plt.plot(x,[y_[2] for y_ in y],label='blue overlapping', color = 'blue',linestyle='dotted')
#x,y = bad_integration_method(gm,start,end)
_f = lambda x: bad_integration_method_(gm,start,x)
y = list(map(_f,x))
plt.plot(x,[y_[0] for y_ in y],label='red no overlapping',linestyle='dashed', color = 'red')
plt.plot(x,[y_[1] for y_ in y],label='green no overlapping',linestyle='dashed', color = 'green')
plt.plot(x,[y_[2] for y_ in y],label='blue no overlapping',linestyle='dashed', color = 'blue')
print("base error no rescaling red",np.abs(y[len(y)-1][0]-quad(f_0,start,end)[0])/quad(f_0,start,end)[0])
print("base error no rescaling green",np.abs(y[len(y)-1][1]-quad(f_1,start,end)[0])/quad(f_1,start,end)[0])
print("base error no rescaling blue",np.abs(y[len(y)-1][2]-quad(f_2,start,end)[0])/quad(f_2,start,end)[0])
plt.xlabel('x')
plt.ylabel('color intensity')
plt.title('integrated values from 0 to x')
plt.legend()
plt.show()

#compute the error according to k
ks = [x for x in range(1,100)]
errors = []
true_values = [quad(f_0,start,end)[0],quad(f_1,start,end)[0],quad(f_2,start,end)[0]]
for k in ks:
    #x,y = my_integration_method(gm,start,end,k)
    _y = my_integration_method_(gm,start,end,k)
    #errors.append([np.abs(y[len(y)-1][i]-true_values[i])/true_values[i] for i in range(3)])
    errors.append([np.abs(_y[i]-true_values[i])/true_values[i] for i in range(3)])
plt.plot(ks,[e[0] for e in errors],label='error red', color = 'red')
plt.plot(ks,[e[1] for e in errors],label='error green', color = 'green')
plt.plot(ks,[e[2] for e in errors],label='error blue', color = 'blue')
plt.xlabel('k')
plt.ylabel('error')
plt.title('error according to k')
plt.semilogy()
plt.legend()
plt.show()
# compute the errors according to k and rescaling
rescaling = 1
errors_1 = []
for k in ks:
    #x,y = my_integration_method(gm,start,end,k)
    #errors_1.append([np.abs(y[len(y)-1][i]-true_values[i])/true_values[i] for i in range(3)])
    _y = my_integration_method_(gm,start,end,k)
    errors_1.append([np.abs(_y[i]-true_values[i])/true_values[i] for i in range(3)])
rescaling = 2
errors_2 = []
for k in ks:
    #x,y = my_integration_method(gm,start,end,k)
    #errors_2.append([np.abs(y[len(y)-1][i]-true_values[i])/true_values[i] for i in range(3)])
    _y = my_integration_method_(gm,start,end,k)
    errors_2.append([np.abs(_y[i]-true_values[i])/true_values[i] for i in range(3)])

# make 3 plot one for red , one for green and one for blue
plt.plot(ks,[e[0] for e in errors],label='error no_rescaling')
plt.plot(ks,[e[0] for e in errors_1],label='error rescaling_1',linestyle='dotted')
plt.plot(ks,[e[0] for e in errors_2],label='error rescaling_2',linestyle='dashed')
plt.xlabel('k')
plt.ylabel('error')
plt.title('error according to k red')
plt.semilogy()
plt.legend()
plt.show()
plt.plot(ks,[e[1] for e in errors],label='error no_rescaling')
plt.plot(ks,[e[1] for e in errors_1],label='error rescaling_1',linestyle='dotted')
plt.plot(ks,[e[1] for e in errors_2],label='error rescaling_2',linestyle='dashed')
plt.xlabel('k')
plt.ylabel('error')
plt.title('error according to k green')
plt.semilogy()
plt.legend()
plt.show()
plt.plot(ks,[e[2] for e in errors],label='error no_rescaling')
plt.plot(ks,[e[2] for e in errors_1],label='error rescaling_1',linestyle='dotted')
plt.plot(ks,[e[2] for e in errors_2],label='error rescaling_2',linestyle='dashed')
plt.xlabel('k')
plt.ylabel('error')
plt.title('error according to k blue')
plt.semilogy()
plt.legend()
plt.show()
print("my error no rescaling red",errors[K-1][0])
print("my error no rescaling green",errors[K-1][1])
print("my error no rescaling blue",errors[K-1][2])
print("my error no rescaling 1 red",errors_1[K-1][0])
print("my error no rescaling 1 green",errors_1[K-1][1])
print("my error no rescaling 1 blue",errors_1[K-1][2])
print("my error no rescaling 2 red",errors_2[K-1][0])
print("my error no rescaling 2 green",errors_2[K-1][1])
print("my error no rescaling 2 blue",errors_2[K-1][2])


#compute the error according to negligeable_val_when_exp
k = K
#negligeable_vals = [-x/50 for x in range(150,500)]
negligeable_vals = np.linspace(-10,0,1000)
errors = []
errors_1 = []
errors_2 = []
rescaling = 0
true_values = [quad(f_0,start,end)[0],quad(f_1,start,end)[0],quad(f_2,start,end)[0]]
for n in negligeable_vals:
    negligeable_val_when_exp = n
    #x,y = my_integration_method(gm,start,end,k)
    rescaling = 0
    _y = my_integration_method_(gm,start,end,k)
    #errors.append([np.abs(y[len(y)-1][i]-true_values[i])/true_values[i] for i in range(3)])
    errors.append([np.abs(_y[i]-true_values[i])/true_values[i] for i in range(3)])
    rescaling = 1
    #x,y = my_integration_method(gm,start,end,k)
    _y = my_integration_method_(gm,start,end,k)
    #errors_1.append([np.abs(y[len(y)-1][i]-true_values[i])/true_values[i] for i in range(3)])
    errors_1.append([np.abs(_y[i]-true_values[i])/true_values[i] for i in range(3)])
    rescaling = 2
    #x,y = my_integration_method(gm,start,end,k)
    _y = my_integration_method_(gm,start,end,k)
    #errors_2.append([np.abs(y[len(y)-1][i]-true_values[i])/true_values[i] for i in range(3)])
    errors_2.append([np.abs(_y[i]-true_values[i])/true_values[i] for i in range(3)])
# make 3 plot one for red , one for green and one for blue
plt.plot(negligeable_vals,[e[0] for e in errors],label='error red no_rescaling')
plt.plot(negligeable_vals,[e[0] for e in errors_1],label='error red rescaling_1',linestyle='dotted')
plt.plot(negligeable_vals,[e[0] for e in errors_2],label='error red rescaling_2',linestyle='dashed')
plt.xlabel('negligeable_val_when_exp')
plt.ylabel('error')
plt.title('error according to negligeable_val_when_exp red')
plt.semilogy()
plt.legend()
plt.show()
plt.plot(negligeable_vals,[e[1] for e in errors],label='error green no_rescaling')
plt.plot(negligeable_vals,[e[1] for e in errors_1],label='error green rescaling_1',linestyle='dotted')
plt.plot(negligeable_vals,[e[1] for e in errors_2],label='error green rescaling_2',linestyle='dashed')
plt.xlabel('negligeable_val_when_exp')
plt.ylabel('error')
plt.title('error according to negligeable_val_when_exp green')
plt.semilogy()
plt.legend()
plt.show()
plt.plot(negligeable_vals,[e[2] for e in errors],label='error blue no_rescaling')
plt.plot(negligeable_vals,[e[2] for e in errors_1],label='error blue rescaling_1',linestyle='dotted')
plt.plot(negligeable_vals,[e[2] for e in errors_2],label='error blue rescaling_2',linestyle='dashed')
plt.xlabel('negligeable_val_when_exp')
plt.ylabel('error')
plt.title('error according to negligeable_val_when_exp blue')
plt.semilogy()
plt.legend()
plt.show()

