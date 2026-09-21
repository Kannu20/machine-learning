# Python Basics

l = 10

print(id(l))

print(True * False + 10)                 

# LIST 
List1 = [1, 2, 3, 4, "Kanishak", 5, 6, 'Todwal', True, 3+4j]
print(List1)

List1.append(7)
print(List1)

List1.remove('Todwal')
print(List1)

print(type(List1))

# TUPLE
Tuple1 = (1, 2, 3, 4, "Kanishak" , 5, 6, 'Todwal', True, 3+4j)
print(Tuple1)
print(type(Tuple1))

# Tuple1.append(6) # This will give an error because tuples are immutable and do not support the append method.

# RANGE

r = range(5, 20)

print(list(r))

# SET

Set1 = set()
print(type(Set1))
print(Set1)
Set1.add(1)
print(Set1)

Set2 = {1, 2,"Sai",3, 4, 5, 6, 7, 8, 9}
print(Set2)

d = {1: "sai", 2: "mohan", 'a': 'apple', 3: 34.5, 1: "mohan"}
print(d)

# d1 = {[1,3,2]: "sai", 2: "mohan", 'a': 'apple', 3: 34.5, 1: "mohan"} # This will give an error because lists are mutable and cannot be used as dictionary keys.
d2 = {(1,3,2): "sai", 2: "mohan", 'a': 'apple', 3: 34.5, 1: "mohan"} # This will work because tuples are immutable and can be used as dictionary keys.
print(d2)

# Indexing and Slicing
name = "Kanishak"
print(name[1:-2])
print(name[1:5])

text = "Python is a powerful programming language"

index = text.find("ow")

print(index)

first = text.find("a")
second = text.find("a", first + 1)
third = text.find("a", second + 1)
fourth = text.find("a", third + 1)
fifth = text.find("a", fourth + 1)

print(first)
print(second)
print(third)
print(fourth)
print(fifth)

s = "Learning Python is very very easy!!!"
print(s[1:-11])
print(s[1:11])
print(s[21:])
print(s[:11])

list1 = [1,3,4,5,7,8,10]

print(list1[1:5])

tuple1 = (1,3,4,5,7,8,10)

print(tuple1[1:5]) 

r = range(1, 10)

print(list(r[2:11:2]))

# Type Casting

# print(complex(10, 20))

print(bool(3 + 2j)) # if the value is greater than and less than 0 then boolean always print True
print(bool(" "))
print(bool("")) # this is empty string so it will give false

print(str(10))
print(str(10.5))
print(str(10 +3j))
print(str(True))
print(str(False))

print('10')

# String Formatting


name = "Kanishak"
age = 22

print(f"My name is {name} and my age is {age}")