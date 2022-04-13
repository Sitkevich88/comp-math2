# 11 вариант
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import *

epsilon = 0.01
x_arr = np.arange(-4, 4, 0.01)
y_arr = []
x = Symbol('x')
y = x

zero_equation = [
    atan(x - 1),
    x * cos(x),
    3 * 2 ** (-(x ** 2 / 20)) - 0.5,
    4.45 * (x ** 3) + 7.81 * (x ** 2) - 9.62 * x - 8.17,
    5 * sin(3 * x - 1)
]


def get_y(x_value):
    return y.evalf(subs={x: x_value})


def get_y_array(x_array, y_sympy):
    res = []
    for x_i in np.nditer(x_array):
        res.append(y_sympy.evalf(subs={x: x_i.item(0)}))
    return np.array(res)


def draw_graph(intervals, x_array, y_array):
    draw_dots(intervals)
    plt.plot(x_array, y_array, label='Выбранное уравнение')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.grid(True)
    plt.legend(loc='upper center')
    plt.show()


def draw_dots(intervals):
    dots_x = (intervals.iloc[:3]['from'] + intervals.iloc[:3]['to']) / 2
    dots_y = np.zeros_like(dots_x)
    plt.scatter(dots_x, dots_y, marker="o", color="red")


def get_intervals(x_array, y_array):
    intervals = pd.DataFrame(data={'from': [], 'to': []})
    skip = False
    for i in range(1, len(x_array) - 2):
        if skip:
            skip = False
            continue

        if (y_array[i] == 0 or (y_array[i - 1] < 0) and (y_array[i + 1] > 0)) or (
                (y_array[i - 1] > 0) and (y_array[i + 1] < 0)):
            if (y_array[i + 1] == 0 or ((y_array[i] < 0) and (y_array[i + 2] > 0))
                or ((y_array[i] > 0) and (y_array[i + 2] < 0))) \
                    and (abs(y_array[i]) > abs(y_array[i + 1])):
                intervals.loc[intervals.shape[0]] = [x_array[i], x_array[i + 2]]
            else:
                intervals.loc[intervals.shape[0]] = [x_array[i - 1], x_array[i + 1]]
            skip = True
    return intervals


def solve(intervals, y_function):
    dots_number = intervals.shape[0]
    res = []
    if dots_number > 2:
        res.append(chord_method(intervals.loc[2]))
    if dots_number > 1:
        res.append(iteration_method(intervals.loc[1], y_function))
    if dots_number > 0:
        res.append(half_division_method(intervals.loc[0]))
    return res


def chord_method(interval):
    data_frame = pd.DataFrame(data={'a': [], 'b': [], 'x': [],
                                    'f(a)': [], 'f(b)': [], 'f(x)': [], '|a-b|': []})
    a = interval['from']
    b = interval['to']
    f_a = get_y(a)
    f_b = get_y(b)

    _x = (a * f_b - b * f_a) / (f_b - f_a)
    f_x = get_y(_x)

    data_frame.loc[data_frame.shape[0]] = [a, b, _x, f_a, f_b, f_x, abs(a - b)]

    while (data_frame.loc[data_frame.shape[0] - 1].iat[6]) > epsilon:
        if (f_a <= 0) and (f_b >= 0):
            if f_x >= 0:
                b = _x
                f_b = f_x
            else:
                a = _x
                f_a = f_x
        else:
            if f_x >= 0:
                a = _x
                f_a = f_x
            else:
                b = _x
                f_b = f_x
        _x = (a * get_y(b) - b * get_y(a)) / (get_y(b) - get_y(a))
        f_x = get_y(_x)
        data_frame.loc[data_frame.shape[0]] = [a, b, _x, f_a, f_b, f_x, abs(a - b)]
    #data_frame.to_csv('chord.csv')
    return data_frame


def iteration_method(interval, y_function):
    data_frame = pd.DataFrame(data={'x_k': [], 'x_(k+1)': [], 'phi(x_(k+1))': [],
                                    'f(x_(k+1))': [], '|x_(k+1) - x_k)|': []})

    a = interval['from']
    b = interval['to']
    y_derivative = y_function.diff(x)
    _lambda = -1 / Max((y_derivative.evalf(subs={x: a})),
                       (y_derivative.evalf(subs={x: b})))
    phi_function = x + (_lambda * y_function)

    x_k = a
    x_k1 = phi_function.evalf(subs={x: x_k})
    phi_x_k1 = phi_function.evalf(subs={x: x_k1})
    f_x_k1 = get_y(x_k1)
    _delta = abs(x_k1 - x_k)
    data_frame.loc[data_frame.shape[0]] = [x_k, x_k1, phi_x_k1, f_x_k1, _delta]
    while _delta > epsilon:
        x_k = x_k1
        x_k1 = phi_function.evalf(subs={x: x_k})
        phi_x_k1 = phi_function.evalf(subs={x: x_k1})
        f_x_k1 = get_y(x_k1)
        _delta = abs(x_k1 - x_k)
        data_frame.loc[data_frame.shape[0]] = [x_k, x_k1, phi_x_k1, f_x_k1, _delta]
    #data_frame.to_csv('iteration.csv')
    return data_frame


def half_division_method(interval):
    data_frame = pd.DataFrame(data={'a': [], 'b': [], 'x': [],
                                    'f(a)': [], 'f(b)': [], 'f(x)': [], '|a-b|': []})
    a = interval['from']
    b = interval['to']
    f_a = get_y(a)
    f_b = get_y(b)

    _x = (a + b) / 2
    f_x = get_y(_x)

    data_frame.loc[data_frame.shape[0]] = [a, b, _x, f_a, f_b, f_x, abs(a - b)]

    while (data_frame.loc[data_frame.shape[0] - 1].iat[6]) > epsilon:
        if (f_a <= 0) and (f_b >= 0):
            if f_x >= 0:
                b = _x
                f_b = f_x
            else:
                a = _x
                f_a = f_x
        else:
            if f_x >= 0:
                a = _x
                f_a = f_x
            else:
                b = _x
                f_b = f_x
        _x = (a + b) / 2
        f_x = get_y(_x)
        data_frame.loc[data_frame.shape[0]] = [a, b, _x, f_a, f_b, f_x, abs(a - b)]
    #data_frame.to_csv('half.csv')
    return data_frame


def get_equation():
    num = -2
    while True:
        print("Выберите номер уравнения")
        for index, value in enumerate(zero_equation):
            print(index, ') ', str(value), "= 0")
        try:
            num = int(input())
        except:
            print("Неверный ввод")
        if 0 <= num < (len(zero_equation)):
            break

    return zero_equation[num]


def get_bounds():
    bounds_ = [-1, -2]
    while True:
        try:
            begin, end = map(int, input("Введите границы \"a b\": ").split())
            if end > begin:
                bounds_[0] = begin
                bounds_[1] = end
                break
            else:
                print("Некорректные границы")
        except:
            print("Некорректный ввод")

    return bounds_


def system_iterations():
    x1 = Symbol('x1')
    x2 = Symbol('x2')

    print("Идёт решение системы")
    y1 = x1 ** 2 - 5 * x1 - x2 ** 2 - 15
    y2 = x2 ** 2 + (x1 ** 2) / 8 + x2 - 17

    point_x1 = 8
    point_x2 = 2

    lambda1 = -1 / (y1.diff(x1).evalf(subs={x1: point_x1, x2: point_x2}))
    lambda2 = -1 / (y2.diff(x2).evalf(subs={x1: point_x1, x2: point_x2}))

    x1_function = x1 + lambda1 * y1
    x2_function = x2 + lambda2 * y2

    check = abs(x1_function.diff(x1).evalf(subs={x1: point_x1, x2: point_x2})) \
        + abs(x1_function.diff(x2).evalf(subs={x1: point_x1, x2: point_x2}))
    if check >= 1:
        print(y1, "= 0", "не сходится")
        return
    check = abs(x2_function.diff(x1).evalf(subs={x1: point_x1, x2: point_x2})) \
        + abs(x2_function.diff(x2).evalf(subs={x1: point_x1, x2: point_x2}))
    if check >= 1:
        print(y2, "= 0", "не сходится")
        return
    should_do = True
    iterations_number = 0
    while should_do:
        a = x1_function.evalf(subs={x1: point_x1, x2: point_x2})
        b = x2_function.evalf(subs={x1: point_x1, x2: point_x2})
        if abs(a-point_x1) <= epsilon and abs(b-point_x2) <= epsilon:
            should_do = False
        point_x1 = a
        point_x2 = b
        iterations_number += 1
    print("Вектор неизвестных\nx1 =", point_x1, ", x2 =", point_x2)
    print("Количество итераций:", iterations_number)
    print("Подставим в 1", y1.evalf(subs={x1: point_x1, x2: point_x2}))
    print("Подставим в 2", y2.evalf(subs={x1: point_x1, x2: point_x2}))
    x1_arr_ = np.arange(7.11, float(str(point_x1))+3, 0.01)
    y1_arr_ = []
    y2_arr_ = []
    for x_i in x1_arr_:
        y1_arr_.append(float(str((sqrt(x1 ** 2 - 5 * x1 - 15)).evalf(subs={x1: x_i}))))
        y2_arr_.append(float(str(((sqrt(17 + 1 / 4 - x1 ** 2 / 8)) - 1 / 2).evalf(subs={x1: x_i}))))
    y1_arr_ = np.array(y1_arr_)
    y2_arr_ = np.array(y2_arr_)

    plt.plot(x1_arr_, y1_arr_, label='Уравнение 1 из системы')
    plt.plot(x1_arr_, y2_arr_, label='Уравнение 2 из системы')
    plt.scatter(float(str(point_x1)), float(str(point_x2)), marker="o", color="red")


if __name__ == '__main__':
    y = get_equation()
    bounds = get_bounds()
    x_arr = np.arange(bounds[0], bounds[1], 0.01)
    y_arr = get_y_array(x_arr, y)
    dots_intervals = get_intervals(x_arr, y_arr)
    results = solve(dots_intervals, y)
    for table in results:
       with pd.option_context('display.max_rows', None, 'display.max_columns', None):
           print(table)
    system_iterations()
    draw_graph(dots_intervals, x_arr, y_arr)
