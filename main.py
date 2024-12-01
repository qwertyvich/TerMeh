import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

t = sp.Symbol('t')
r = 1 + 1.5 * sp.sin(12 * t)
phi = 1.2 * t + 0.2 * sp.cos(12 * t)
x = r * sp.cos(phi)
y = r * sp.sin(phi)

Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)
V = sp.sqrt(Vx**2 + Vy**2)                  
kappa = sp.Abs(Vx * Ay - Vy * Ax) / V**3    
rho = 1 / kappa                             

F_x = sp.lambdify(t, x, modules='numpy')
F_y = sp.lambdify(t, y, modules='numpy')
F_Vx = sp.lambdify(t, Vx, modules='numpy')
F_Vy = sp.lambdify(t, Vy, modules='numpy')
F_Ax = sp.lambdify(t, Ax, modules='numpy')
F_Ay = sp.lambdify(t, Ay, modules='numpy')
F_rho = sp.lambdify(t, rho, modules='numpy')

t_vals = np.linspace(0, 2 * np.pi, 1000)    

x_vals = F_x(t_vals)
y_vals = F_y(t_vals)
Vx_vals = F_Vx(t_vals)
Vy_vals = F_Vy(t_vals)
Ax_vals = F_Ax(t_vals)
Ay_vals = F_Ay(t_vals)
rho_vals = F_rho(t_vals)

Alpha_V = np.arctan2(Vy_vals, Vx_vals) 
Alpha_A = np.arctan2(Ay_vals, Ax_vals)

fig, ax = plt.subplots(figsize=(12, 12))
ax.axis('equal')
ax.set_xlim(-8, 8)                      
ax.set_ylim(-8, 8)                      
ax.grid(True)                           
ax.plot(x_vals, y_vals, label='Траектория')

P, = ax.plot([], [], 'ro', label='Точка')
V_line, = ax.plot([], [], color='red', label='Скорость') 
V_arrow, = ax.plot([], [], color='red')                     
A_line, = ax.plot([], [], color='green', label='Ускорение')     
A_arrow, = ax.plot([], [], color='green')                           
circle = plt.Circle((0, 0), 0, color='purple', fill=False, label='Радиус кривизны')
ax.add_patch(circle)

a = 0.05                                    
b = 0.025                                   
x_arr = np.array([-a, 0, -a])
y_arr = np.array([b, 0, -b])
k_V = 0.2 
k_A = 0.05 
max_rho = 6     
max_center_dist = 6 

def Rot2D(X, Y, Alpha):
    RotX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RotY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RotX, RotY

def animate(i):
    P.set_data(x_vals[i], y_vals[i])    

    V_end_x = x_vals[i] + k_V * Vx_vals[i]                      
    V_end_y = y_vals[i] + k_V * Vy_vals[i]
    V_line.set_data([x_vals[i], V_end_x], [y_vals[i], V_end_y])
    RotX_V, RotY_V = Rot2D(x_arr, y_arr, Alpha_V[i])
    V_arrow.set_data(V_end_x + RotX_V, V_end_y + RotY_V)

    A_end_x = x_vals[i] + k_A * Ax_vals[i]                              
    A_end_y = y_vals[i] + k_A * Ay_vals[i]
    A_line.set_data([x_vals[i], A_end_x], [y_vals[i], A_end_y])
    RotX_A, RotY_A = Rot2D(x_arr, y_arr, Alpha_A[i])
    A_arrow.set_data(A_end_x + RotX_A, A_end_y + RotY_A)

    V_mag = np.sqrt(Vx_vals[i]**2 + Vy_vals[i]**2)
    if V_mag == 0:
        circle.set_visible(False)
        return P, V_line, V_arrow, A_line, A_arrow, circle

    Nx = -Vy_vals[i] / V_mag
    Ny = Vx_vals[i] / V_mag
    x_c = x_vals[i] + Nx * rho_vals[i]
    y_c = y_vals[i] + Ny * rho_vals[i]

    center_dist = np.sqrt((x_c - x_vals[i])**2 + (y_c - y_vals[i])**2)

    if rho_vals[i] <= max_rho and center_dist <= max_center_dist:
        circle.set_visible(True)
        circle.center = (x_c, y_c)
        circle.radius = rho_vals[i]
    else:
        circle.set_visible(False)

    return P, V_line, V_arrow, A_line, A_arrow, circle

ani = FuncAnimation(fig, animate, frames=len(t_vals), interval=20, blit=True)

ax.legend()
plt.show()