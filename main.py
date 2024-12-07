import numpy as np                                      # библитека для мат операций
import matplotlib.pyplot as plt                         # для построения графиков
from matplotlib.animation import FuncAnimation          # для создания анимации
import sympy as sp                                      # для символьных вычислений

#опроделяю символьную перменную имеющую алгебраичский смысл
t = sp.Symbol('t')

# задаю радиус как функцию времени с гармоническим отклонением
r = 1 + 1.5 * sp.sin(12 * t)

# задаю угол φ как линейную функцию времени с гармоническим отклонением
phi = 1.2 * t + 0.2 * sp.cos(12 * t)

# определяю координаты x и y в полярных координатах
x = r * sp.cos(phi)
y = r * sp.sin(phi)

# вычисляю производные координат для получения компонент скорости
Vx = sp.diff(x, t)              # f'(x) по времени
Vy = sp.diff(y, t)                  # f'(y) по времени

# вычисляю вторые производные для получения компонент ускорения
Ax = sp.diff(Vx, t)             # производная Vx по времени
Ay = sp.diff(Vy, t)             #   производная Vy по времени

# вычстляю величину скорости
V = sp.sqrt(Vx**2 + Vy**2)                  

# вычисляю кривизну траектории
kappa = sp.Abs(Vx * Ay - Vy * Ax) / V**3    

# вычисляю радиус кривизны
rho = 1 / kappa                             

# преобразовал символьные выражения в числовые функции для дальнейшего использования с numpy
F_x = sp.lambdify(t, x, modules='numpy')                # F x(t)
F_y = sp.lambdify(t, y, modules='numpy')                # F y(t)
F_Vx = sp.lambdify(t, Vx, modules='numpy')              # F Vx(t)
F_Vy = sp.lambdify(t, Vy, modules='numpy')              # F Vy(t)
F_Ax = sp.lambdify(t, Ax, modules='numpy')              # F Ax(t)
F_Ay = sp.lambdify(t, Ay, modules='numpy')              # F Ay(t)
F_rho = sp.lambdify(t, rho, modules='numpy')            # F rho(t)

# создал массив значений времени от 0 до 2pi с 1000 точками
t_vals = np.linspace(0, 2 * np.pi, 1000)    

# вычисляю значения координат, скорости, ускорения и радиуса кривизны для всех точек времени
x_vals = F_x(t_vals)
y_vals = F_y(t_vals)
Vx_vals = F_Vx(t_vals)
Vy_vals = F_Vy(t_vals)
Ax_vals = F_Ax(t_vals)
Ay_vals = F_Ay(t_vals)
rho_vals = F_rho(t_vals)

# вычислил углы направлений вектора скорости и ускорения
Alpha_V = np.arctan2(Vy_vals, Vx_vals)  # угол вектора скорости
Alpha_A = np.arctan2(Ay_vals, Ax_vals)  # угол вектора ускорения

# настройка графика
fig, ax = plt.subplots(figsize=(12, 12))  # созлал фигуру и оси с размером 12x12 дюймов
ax.axis('equal')  # установил равные масштабы по осям
ax.set_xlim(-8, 8)  # установил пределы по оси X
ax.set_ylim(-8, 8)  # установил пределы по оси Y
ax.grid(True)  # вылючил сетку на графике
ax.plot(x_vals, y_vals, label='Траектория')  # нарисовал траекторию движения

# задал объекты для анимации
P, = ax.plot([], [], 'ro', label='Точка')  # точка на траектории
V_line, = ax.plot([], [], color='red', label='Скорость')  # линия - вектор скорости
V_arrow, = ax.plot([], [], color='red')  # вектора скорости
A_line, = ax.plot([], [], color='green', label='Ускорение')  # вектор ускорения
A_arrow, = ax.plot([], [], color='green')  # вектор ускорения
circle = plt.Circle((0, 0), 0, color='purple', fill=False, label='Радиус кривизны')  # радиус кривизны
ax.add_patch(circle)  # доабвил круг на график

# настройка для стрелок
a = 0.05            # полуширина стрелки
b = 0.025               # полувысота стрелки
x_arr = np.array([-a, 0, -a])           #  X для формы стрелки
y_arr = np.array([b, 0, -b])            #  Y для формы стрелки
k_V = 0.2                               # Масштаб для вектора скорости
k_A = 0.05                          # масштаб для вектора ускорения
max_rho = 6                         # максимальный радиус кривизны для отображения
max_center_dist = 6                     # максимальное расстояние до центра кривизны для отображения

# F поворота координат на заданный угол
def Rot2D(X, Y, Alpha):
    RotX = X * np.cos(Alpha) - Y * np.sin(Alpha)            # поворот координаты X
    RotY = X * np.sin(Alpha) + Y * np.cos(Alpha)            # поворот координаты Y
    return RotX, RotY

#функция обновления кадров анимации
def animate(i):
    P.set_data(x_vals[i], y_vals[i])            # обновил положение точки P
    #   вычисляю конец вектора скорости
    V_end_x = x_vals[i] + k_V * Vx_vals[i]                      
    V_end_y = y_vals[i] + k_V * Vy_vals[i]
    V_line.set_data([x_vals[i], V_end_x], [y_vals[i], V_end_y])  # обновил линию скорости
    
    # повернул стрелку скорости в нужном направлении
    RotX_V, RotY_V = Rot2D(x_arr, y_arr, Alpha_V[i])
    V_arrow.set_data(V_end_x + RotX_V, V_end_y + RotY_V)  # обнвоил стрелку скорости
    
    # вычислил конец вектора ускорения
    A_end_x = x_vals[i] + k_A * Ax_vals[i]                              
    A_end_y = y_vals[i] + k_A * Ay_vals[i]
    A_line.set_data([x_vals[i], A_end_x], [y_vals[i], A_end_y])  # обновил линию ускорения
    
    # повернул стрелку ускорения в нужном направлении
    RotX_A, RotY_A = Rot2D(x_arr, y_arr, Alpha_A[i])
    A_arrow.set_data(A_end_x + RotX_A, A_end_y + RotY_A)  # обнвоил стрелку ускорения
    
    # вычтялил величину скорости
    V_mag = np.sqrt(Vx_vals[i]**2 + Vy_vals[i]**2)
    if V_mag == 0:
        circle.set_visible(False)  # скрываю круг радиуса кривизны если v=0
        return P, V_line, V_arrow, A_line, A_arrow, circle
    
    # вычислил нормаль к траектории
    Nx = -Vy_vals[i] / V_mag
    Ny = Vx_vals[i] / V_mag
    # нашёл центр кривизны
    x_c = x_vals[i] + Nx * rho_vals[i]
    y_c = y_vals[i] + Ny * rho_vals[i]
    
    # вычислил расстояние до центра кривизны
    center_dist = np.sqrt((x_c - x_vals[i])**2 + (y_c - y_vals[i])**2)
    
    # проверил условия отображения круга радиуса кривизны
    if rho_vals[i] <= max_rho and center_dist <= max_center_dist:
        circle.set_visible(True)                             # оторбразил круг
        circle.center = (x_c, y_c)                            # установил центр круга
        circle.radius = rho_vals[i]                           # устанвоил радиус круга
    else:
        circle.set_visible(False)                   # скрыл круг, если условия не выполнены
    
    # вернул обновлённые объекты для перерисовки
    return P, V_line, V_arrow, A_line, A_arrow, circle

# создал анимацию с заданными параметрами
ani = FuncAnimation(fig, animate, frames=len(t_vals), interval=20, blit=True)

# добавил описание к графику
ax.legend()

# вывел график
plt.show()
