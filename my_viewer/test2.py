import numpy as np
a  = -10
b = 20
T_a = 1
def erf(x):
    return np.sqrt(1-np.exp(x**(2)*((-1.27324-0.074647* x**(2))/(1+0.0886745* x**(2)))))
T_b1 = T_a*np.exp(erf(b)-erf(a))
v1 = T_a*(1-np.exp(erf(b)-erf(a)))

k = 100
v2=0
T = T_a
for i in range(k):
    x = a + (b-a)*i/k
    y = x + (b-a)/k
    v2+= T * (1-np.exp(erf(y)-erf(x)))
    T = T * np.exp(erf(y)-erf(x))
T_b2 = T
print(v1,v2,T_b1,T_b2)