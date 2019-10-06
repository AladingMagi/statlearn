from scipy.optimize import minimize
import numpy as np
# 目标函数
fun = lambda x : 10 - x[0]**2 - x[1]**2
# 约束条件
cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] }, # x1+x2=0
        {'type': 'ineq', 'fun': lambda x: x[1] - x[0]**2} # x2-x1^2>=0
       )
# 设置初始值
x0 = np.array((-0.2, 0.2))
res = minimize(fun, x0, method='SLSQP', constraints=cons)
print('目标函数最小值：',res.fun)
print('目标函数最优解[x1, x2]：',res.x)
print('迭代终止是否成功：', res.success)
print('迭代终止原因：', res.message) 

