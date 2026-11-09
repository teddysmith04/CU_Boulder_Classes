"""
Solve a system of equations for a 2D grid of values. Given equations for dopler shift, and apparent velocity, values for doppler shift and apparaent velocity, find actual velocity and viewing angle:
"""

import numpy as np
from scipy.optimize import fsolve

### Define funcitons:
def Vpm(vector):
    V, theta = vector
    """
    Function to solve for the apparent velocity given actual velocity and viewing angle
    """
    return V * np.sin(theta) / (1 + V * np.cos(theta))

def doppler(vector):
    V, theta = vector
    """
    Function to solve for the doppler shift given actual velocity and viewing angle
    """
    return (1 + (V * np.cos(theta))) / (np.sqrt(1 - V**2))

def composite(vector):
    V, theta = vector
    """
    Function to solve for both apparent velocity and doppler shift given actual velocity and viewing angle
    """
    return [Vpm(vector) - Vpm_condtition, doppler(vector) - doppler_condition]


# ### Define given values:
Vpm_condtition = 2.52272
doppler_condition = 0.343035

### Solve for V and theta using fsolve:
solutions = fsolve(composite, [0.5, 0.5])
print(f"Actual velocity (V): {solutions[0]}")
print(f"Viewing angle (theta in deg): {(solutions[1] * 180 / np.pi) % 180}")

## sanity check 
print(np.isclose(composite(solutions), [0, 0]))




##### Attempt 2
v_grid = np.linspace(0.01, 0.99, 1000)  # Actual velocity grid
theta_grid = np.linspace(0.01, np.pi, 1000)  # Viewing angle grid



### find where close w np.where
# solutions = [] 
# for v in v_grid:
#     for theta in theta_grid:
#         if (np.abs(Vpm([v, theta]) - Vpm_condtition) < 1e-3) and (np.abs(doppler([v, theta]) - doppler_condition) < 1e-3):
#             solutions.append([v, theta])

# ### Take median solutions:
# median_v = np.median([s[0] for s in solutions])
# median_T = np.median([s[1] for s in solutions])

# print(f"Actual velocity (V): {median_v}")
# print(f"Viewing angle (theta in deg): {median_T * 180 / np.pi}")

