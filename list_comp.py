#l1 = input("Fill the first list")
#l2 = input("Fill the second list")

l1 = ['white','green','brown','red']
l2 = ['brown','green','red','blue']


not_there= []

s1 ='white'
s2 = 'white'

for word1 in l1:
    if word1 not in l2:
        not_there = str(not_there) + word1

for word1 in l2:
    if word1 not in l1:
        not_there = str(not_there) + " " + word1

print(not_there)
