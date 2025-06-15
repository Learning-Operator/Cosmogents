

To solve ordinary differential equations, you can use the scipy ordinary differential equation solver.

to use use the scipy diary, you can use:

import scipy

Here is the definition of the functions you can use to solve ordinaty differential equations with the Scipy library.


# scipy.integrate.solve_ivp

solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, **options)

Where:
- fun is the right-hand side of the system
- t_span is the interval of time you are going over
- y0 is an array of shape (n,) that is the initial state
- 'method' is a string, and is optional
- t_eval is the times at which to store the computed solutions
- dense_output is a boolean variable which determines wether to compute continuous solutions
- events is a variable where you specify events to track

you can also call these optional variables:

- first_step is the initial step size
max_step is the maximum allowed step size


methods that can be:
- 'RK45' --> the explicit Runge-Kutta method of order 5
- 'RK23' --> the explicit Runge-Kutta method of order 3
- 'DOP853' --> the explicit Runge-Kutta method of order 8
- 'Radau' --> the implicit Runge-Kutta method of order 5
- 'BDF' --> the Implicit multi-step variable-order (1 to 5) method based on a backward differentiation formula for the derivative approximation
- 'LSODA' --> this is a wrapper of the Fortran solver from ODEPACK.



## an example use:

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def dydt(t, y):
    # Example: simple harmonic oscillator y = [x, v]
    x, v = y
    return [v, -x]

t_span = (0, 10)
y0 = [1.0, 0.0]        # start at x=1, v=0
t_eval = np.linspace(0, 10, 200)

sol = solve_ivp(
    fun=dydt,
    t_span=t_span,
    y0=y0,
    method='RK45',
    t_eval=t_eval,
    max_step=0.1
)

plt.plot(sol.t, sol.y[0])
plt.xlabel('t'); plt.ylabel('x(t)')
plt.title('Simple Harmonic Oscillator')
plt.show()

