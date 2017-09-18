import random

y = [2,5,8,10,16,18,40,42,25,28,30,15]
x= [2,5,8,10,7,3,6]
s = random.Random(622).sample(y,k=5)
l = random.Random(528).sample(x,k=1)
print(s,l)