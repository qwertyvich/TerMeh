import numpy as np           # для работы с массивами
import matplotlib.pyplot as plt  # ля построения графиков (matplotlib)
from matplotlib.animation import FuncAnimation  #для создания анимаций      
import math                

#функция для поворота точки на плоскости
def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)  # поворот по оси X
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)  # поворот по оси Y
    return RX, RY  # возвращаем новые координаты после поворота

# определяю константы и функции для угловых перемещений
R = 2           # радиус диска
a = 1           # расстояние от центра O до точки A
l = 2.5           # ллина стержня AB
phi_t = np.sin  # функция углового перемещения \phi(t)
psi_t = np.cos  # функция углового перемещения \psi(t)

steps = 500
T = np.linspace(0, 40, steps)  # массив времени от 0 до 40 с 500 шагами

#функция для вычисления координат точек в зависимости от времени
def compute_positions(t):
    phi = phi_t(t)  # угол для точки A
    psi = psi_t(t)  # угол для точки B
    
    #позиция точки O (центр диска, фиксированная)
    X_O, Y_O = 0, 0
    
    #позиция точки A (на диске)
    X_A = X_O + a * np.sin(phi)
    Y_A = Y_O - a * np.cos(phi)
    
    #позиция точки B (конец стержня AB)
    X_B = X_A - l * np.sin(psi)
    Y_B = Y_A - l * np.cos(psi)
    
    return X_O, Y_O, X_A, Y_A, X_B, Y_B  #возвращаем координаты всех точек

#подготовим массивы для хранения позиций
X_O, Y_O = [], []  #списки для хранения координат точки O
X_A, Y_A = [], []  #списки для хранения координат точки A
X_B, Y_B = [], []  #списки для хранения координат точки B

#для каждого времени из массива T вычисляем позиции
for t in T:
    xo, yo, xa, ya, xb, yb = compute_positions(t)
    X_O.append(xo)  #добавляю координаты точки O в список
    Y_O.append(yo)  #добавляю координаты точки O в список
    X_A.append(xa)  #добавляю координаты точки A в список
    Y_A.append(ya)  #добавляю координаты точки A в список
    X_B.append(xb)  #добавляю координаты точки B в список
    Y_B.append(yb)  #добавляю координаты точки B в список

#настраиваю график
fig, ax = plt.subplots(figsize=(8, 8))  #создаю фигуру и оси для графика
ax.axis('equal')  #устанавливаю равные масштабы по осям
ax.set_xlim(-R - l, R + l)  #устанавливаю пределы оси X
ax.set_ylim(-R - l, R + l)  #устанавливаю пределы оси Y
ax.set_xlabel("x")  # подписываб ось X
ax.set_ylabel("y")  # подписываю ось Y

#строю диск, стержень и точки
phi = np.linspace(0, 2 * np.pi, 100)  #угол для рисования окружности
Disk, = ax.plot(R * np.cos(phi), R * np.sin(phi), 'gray')  #рисую диск серым цветом
Rod, = ax.plot([], [], 'b-', linewidth=2)  #Стержень, который будет обновляться
Point_O, = ax.plot([], [], 'ro', markersize=8)  #центр диска)
Point_A, = ax.plot([], [], 'go', markersize=8)  # Точка A
Point_B, = ax.plot([], [], 'mo', markersize=8)  # Точка B

#обновление анимации
def animate(i):
    # обновляю позиций на графике
    Disk.set_data(R * np.cos(phi) + X_O[i], R * np.sin(phi) + Y_O[i])  # обноволяю диск
    Rod.set_data([X_A[i], X_B[i]], [Y_A[i], Y_B[i]])  # обнволяю стержень
    Point_O.set_data([X_O[i]], [Y_O[i]])  # обноволяю точку O
    Point_A.set_data([X_A[i]], [Y_A[i]])  # обноволяю точку A
    Point_B.set_data([X_B[i]], [Y_B[i]])  # обноволяю точку B
    return Disk, Rod, Point_O, Point_A, Point_B  # вернул обновленные объекты

# Запуск анимации
ani = FuncAnimation(fig, animate, frames=steps, interval=50, blit=True)
plt.show() #вывожу анимацию
