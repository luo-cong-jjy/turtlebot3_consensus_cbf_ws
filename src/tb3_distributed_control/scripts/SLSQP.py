from scipy.optimize import minimize
import numpy as np

# 定义目标函数
def objective_function(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

# 定义约束条件
def constraint(x):
    return 2*x[0] + x[1] - 4

# 初始化参数
x0 = [0.5, 0.5]

# 调用SLSQP算法
result = minimize(objective_function, x0, method='SLSQP', constraints={'type': 'eq', 'fun': constraint})

# 输出最优解
print("最优解：", result.x)