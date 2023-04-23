from collections import Counter


a  = [2,2,3,4,23,1234,1232,3,4,3,3]


x = Counter(a)
#print (x)
top_1 = x.most_common(1)


print (top_1)