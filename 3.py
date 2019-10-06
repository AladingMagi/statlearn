import random
from functools import reduce
NUM=100000
miu1=6
sigma1=0.3
data=[random.gauss(miu1,sigma1) for i in range(NUM)]
miu=reduce(lambda x,y:x+y,data)/NUM
sigma=(reduce(lambda x,y:x+y,map(lambda x:(x-miu1)**2,data))/NUM)**0.5
print('N =',NUM)
print('The truth (µ, σ2):(',miu1,',',sigma1,')')
print('Calculate result:(',miu,',',sigma,')')