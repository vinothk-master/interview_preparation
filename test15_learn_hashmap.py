import time
from collections import defaultdict


str = ['eat', 'tea', 'tan', 'ate', 'nat', 'bat', 'cat', 'mat']


alogram_map =defaultdict(list) #iniializing the keys
result = []

for s in str:
    print(tuple(sorted(s)))
    alogram_map[tuple(sorted(s))].append(s)
print(alogram_map)















#for s in str:
#    sorted_list = tuple(sorted(s))
#    print(sorted_list)
#    alogram_map[sorted_list].append(s)
#    print(alogram_map)
#for count in range(0, 10):
#    print(count)
#
#
#for j in alogram_map.values():
#    result.append(j)
#print("The final result is... :",result)