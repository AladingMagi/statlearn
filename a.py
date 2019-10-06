from sympy import *
k = 1
x1,x2,lam,eta= symbols('x1,x2,l,n', real=True)
obj_func = 10 - x1**2 -x2**2 + lam*(x1+x2) + eta*(x1**2-x2)
L1 = diff(obj_func, x1, k)
L2 = diff(obj_func, x2, k)
L3 = diff(obj_func, lam, k)
L4 = diff(obj_func, eta, k)
L5 = x1+x2
L6 = eta*(x1**2-x2)

res = solve([L1,L2,L3,L4,L5,L6],[x1,x2,lam,eta])
res_x1 = res[0][0]
res_x2 = res[0][1]
res_lam = res[0][2]
res_eta = res[0][3]
res_min = obj_func.subs([(x1,res_x1),(x2,res_x2)])
print('拉格朗日乘子：lambda =',res_lam,',eta =',res_eta)
print('最优解:x1 =',res_x1,',x2 =',res_x2)
print('目标函数最小值:',res_min)

