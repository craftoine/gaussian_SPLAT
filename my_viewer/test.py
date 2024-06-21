#plot the aproximation with tailor serie of error functions for differnt degrees
import numpy as np
import matplotlib.pyplot as plt
import math

def true_error_function(x):
    return np.sqrt(1-np.exp(x**(2)*((-1.27324-0.074647* x**(2))/(1+0.0886745* x**(2)))))
def true_exp_error_function(x):
    return np.exp(true_error_function(x))

def factorial(n):
    k = 1
    for i in range(1,n+1):
        k = k*i
    return k

n = 64
coefficients_error = [0 if i%2 == 0 else (2/np.sqrt(math.pi))*(-1)**((i-1)/2)/(i*factorial((i-1)//2)) for i in range(n)]
coefficients_exp = [1/factorial(i) for i in range(n)]
coefficients_exp_error = [0 for i in range(n**2)]

coefficients_error_power = [[0 for i in range(len(coefficients_exp_error))] for j in range(n)]

coefficients_error_power[0][0] = 1

for i in range(1,n):
    print(i)
    for j in range(n):
        for l in range(len(coefficients_error_power[i])):
            deg = j+l
            if deg < len(coefficients_error_power[i]):
                coefficients_error_power[i][deg] += coefficients_error[j]*coefficients_error_power[i-1][l]
            else:
                break

for i in range(n):
    for j in range(len(coefficients_exp_error)):
        coefficients_exp_error[j] += coefficients_exp[i]*coefficients_error_power[i][j]
print(coefficients_exp_error)
def error_function(x, n):
    f = 0
    for i in range(n):
        f = f + coefficients_error[i]*x**i
    return f

def exp_function(x, n):
    f = 0
    for i in range(n):
        f = f + coefficients_exp[i]*x**i
    return f

def exp_error_function(x, n):
    
    """f = 0
    for i in range(n):
        f = f + coefficients_exp_error[i]*x**i
    return f"""
    #do calculations under a logarithm
    f = 0
    for i in range(n):
        if coefficients_exp_error[i] != 0:
            f = f + np.exp(
                np.log(abs(coefficients_exp_error[i])) + i*np.log(x)
            )*np.sign(coefficients_exp_error[i])

    return f
    
x = np.linspace(0.01,10,1000)
#n_values = [i**2 for i in range(1,int(math.sqrt(n))+1)]
n_values = [2,8,16,32,64]

#plot the error function aproximation
plt.figure()
plt.plot(x, true_error_function(x), label = 'True error function')
for n in n_values:
    plt.plot(x, error_function(x,n), label = 'Aproximation with n = ' + str(n))
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Error function aproximation')
#bound y axis
plt.ylim(0,3)
plt.show()

#plot the exp function aproximation
plt.figure()
plt.plot(x, np.exp(x), label = 'True exp function')
for n in n_values:
    plt.plot(x, exp_function(x,n), label = 'Aproximation with n = ' + str(n))
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Exp function aproximation')
#bound y axis
plt.ylim(0,3*np.exp(1))
plt.show()

#plot the exp error function aproximation
plt.figure()
plt.plot(x, true_exp_error_function(x), label = 'True exp error function')
plt.plot(x, exp_function(error_function(x,n),n), label = 'composition')
for n in n_values:
    plt.plot(x, exp_error_function(x,n**2), label = 'Aproximation with n = ' + str(n**2))
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Exp error function aproximation')
#bound y axis
plt.ylim(0,3*np.exp(1))
plt.show()
