import numpy as np
from scipy.integrate import odeint
from mpmath import *
from math import cosh, sinh, sin, cos, sqrt, pow, pi
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from numba import njit
mp.dps = 3

# dimensions in m
a = 20/1000
b = 20/1000

x_c = a/2
y_c = b/2

c = 2/1000
d = 2/1000

T_f = 20 # in °C
h = 1*pow(10, 4) # example
t1 = 5/1000
k1 = 239 # aluminium

Q_in = 5 # Watt?

_delta = lambda n: n*pi/b
_lambda = lambda m: m*pi/a
_beta = lambda l, d: sqrt(pow(l,2) + pow(d,2))

A_0 = Q_in/(a*b) * (t1/k1 + 1/h) 
B_0 = - Q_in/(k1*a*b)

B_1 = lambda m: -phi(_lambda(m))*A_m(m)
B_2 = lambda n: -phi(_delta(n))*A_n(n)
B_3 = lambda n, m: -phi(_beta(_lambda(m), _delta(n)))*A_mn(n, m) 

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

@np.vectorize
def thetha_xyz(x, y, z):
    print("Calculating  {},{},{}".format(x,y,z))
    
    summand_1 = nsum(lambda m: cos(_lambda(m)*x)*(A_m(m)*cosh(_lambda(m)*z) + B_1(m)*sinh(_lambda(m)*z)), [1, inf])
    summand_2 = nsum(lambda n: cos(_delta(n)*y)*(A_n(n)*cosh(_delta(n)*z) + B_2(n)*sinh(_delta(n)*z)), [1, inf])
    summand_3 = nsum(lambda n, m: cos(_lambda(m)*x)*cos(_delta(n)*y)*(A_mn(n, m)*cosh(_beta(_lambda(m), _delta(n))*z) + B_3(n, m)*sinh(_beta(_lambda(m), _delta(n))*z)), [1, inf], [1, inf])

    ret = A_0 + B_0*z + summand_1 + summand_2 + summand_3
    return float(ret)


def A_1(m):
    ret = (Q_in / (b*c*k1*_lambda(m)*phi(_lambda(m)) )) \
        * (nsum(lambda x: cos(_lambda(m)*x), [x_c- c/2, x_c+ c/2])) / nsum(lambda x: pow(cos(_lambda(m)*x),2), [0, a])
    return ret


def A_2(m):
    ret = (Q_in / (a*d*k1*_delta(m)*phi(_delta(m)) )) \
        * (nsum(lambda y: cos(_delta(m)*y), [y_c- d/2, y_c+ d/2]) / nsum(lambda y: pow(cos(_delta(m)*y),2), [0, b]))
    return ret


def A_3(n,m):
    ret = (Q_in / (c*d*k1*_beta(_lambda(m), _delta(n))*phi(_beta(_lambda(m), _delta(n))) )) \
        * (nsum(lambda x, y: cos(_lambda(m)*x)*cos(_delta(n)*y), [x_c- c/2, x_c+ c/2], [y_c- d/2, y_c+ d/2]) \
        / nsum(lambda x, y: pow(cos(_lambda(m)*x),2) * pow(cos(_delta(m)*y),2), [0, a], [0, b]))
    return ret

def A_mn(n, m):
    ret = (16*Q_in*cos(_lambda(m)*x_c)*sin(0.5*_lambda(m)*c)*cos(_delta(n)*y_c)*sin(0.5*_delta(n)*d)) \
        / (a*b*c*d*k1*_beta(_lambda(m), _delta(n))*_lambda(m)*_delta(n)*phi(_beta(_lambda(m), _delta(n))))
    return ret

def A_m(m):
    ret = (2*Q_in* ( sin(((2*x_c + c)/2)*_lambda(m)) - sin(((2*x_c - c)/2)*_lambda(m))) ) / (a*b*c*k1*(pow(_lambda(m),2))*phi(_lambda(m)))
    return ret

def A_n(m):
    ret = (2*Q_in* ( sin(((2*y_c + d)/2)*_delta(m)) - sin(((2*y_c - d)/2)*_delta(m))) ) / (a*b*d*k1*(pow(_delta(m),2))*phi(_delta(m)))
    return ret

def phi(zeta):
    ret = (zeta*sinh(zeta*t1) + h/k1 * cosh(zeta*t1)) / (zeta*cosh(zeta*t1) + h/k1 * sinh(zeta*t1))
    return ret

if __name__=="__main__":
    # ret = thetha_xyz(0,0,0)
    # print("Temperature in upper corner: " + str(ret) + "°C")
    # ret = thetha_xyz(x_c,y_c,0)
    # print("Temperature at chip-center-interface: " + str(ret) + "°C")
    # ret = thetha_xyz(x_c, y_c, t1)
    # print("Temperature at center-bottom: " + str(ret) + "°C")
    
    nr = 0.001
    x_coords = np.arange(0, a, nr)
    print(x_coords)
    y_coords = np.arange(0, b, nr)
    print(y_coords)
    X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
    fig, axes = plt.subplots(nrows=1, ncols=6)
    fig.set_size_inches(25, 3)

    for nr, x in enumerate([0.0, 0.001, 0.002, 0.003, 0.004, 0.005]):
        Z_mesh = thetha_xyz(X_mesh, Y_mesh, x)
        print(Z_mesh)

        norm = cm.colors.Normalize(vmax=Z_mesh.max(), vmin=0)

        cset1 = axes[nr].contourf(
            X_mesh, Y_mesh, Z_mesh, 40,
            norm=norm)
        axes[nr].set_aspect(1)

    #plt.colorbar(cset1)
    plt.savefig('figure.pdf')
    plt.show()