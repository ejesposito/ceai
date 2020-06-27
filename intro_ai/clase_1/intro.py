

#  Alias
num_1 = 10  # creates a new object "o"
print(hex(id(num_1)))  # print the memory address of the object "o" -> '0x103be3cc0'
num_2 = num_1  # creates an alias to the object "o"
print(hex(id(num_2)))  # '0x103be3cc0'


# Immutable data types
a = 10
print(hex(id(a)))  # 0x103be4800
a = 20
print(hex(id(a)))  # 0x103be5480
                   # another memory address is created because the objetct is immutable
a = a + 1
print(hex(id(a)))  # 0x103be54a0


# Immutable data types in methods
def power(base, exponent):
    base = 20
    print(hex(id(base)))  # 0x103be3e00
    return base ** exponent

a = 10
exponent = 2
print(hex(id(a)))  # 0x103be3cc0
print(power(a, exponent))
print(hex(id(a)))  # 0x103be3cc0


# Everything is an object
# Everything is an object (also the numbers)
z = 201
w = 201
print(hex(id(z)))  # 0x103be54a0
print(hex(id(w)))  # 0x103be54a0


# Tuple
t_1 = (1, ['a', 'b'])
t_2 = t_1
t_2[1].append('c')
print(t_1)  # (1, ['a', 'b', 'c'])
print(t_2)  # (1, ['a', 'b', 'c'])
# t_2[1] = ['d'] not a valid operation


# Mutable types
list_1 = list([1,2,3,4])
list_2 = list_1
print(list_2)  # [1,2,3,4]
print(hex(id(list_1)))  # 0x105f05a00
print(hex(id(list_2)))  # 0x105f05a00
list_1.pop()
list_2.pop()
print(list_1) # [1,2]
print(list_2) # [1,2]
print(hex(id(list_1))) # 0x105f05a00


# List
list_1 = [1,2,3,4]
print(type(list_1))  # <class 'list'>
list_1.append(5)
for elm in list_1:
    print(list_1)


# Dict
dict_1 = {
    "key_1": 1,
    "key_2": 2,
    "key_3": 3
}
print(type(dict_1))  # <class 'dict'>
print(dict_1.get("key_1", None))  # 1
print(dict_1.get("key_4", None))  # None
for key, value in dict_1.items():
    print("{}: {}".format(key, value))


# Set
set_1 = set([1,2,3,4,1])  # <class 'set'>
print(type(set_1))
for item in set_1:
    print(item)


# Flow control
a = True
if a:
    print('Hello world!')

b = False
if a and b:
    print('Hello world!')

c = 1
if a == 1:
    print('1')
elif a == 2:
    print('2')
else:
    print('3')

list_1 = [1,2,3,4,5,6]

for item in list_1:
    print(item)

for i, item in enumerate(list_1):
    print("{}: {}".format(i, item))


# List comprehension
list_1 = [1,2,3,4,5,6]
list_2 = [item for item in list_1]
print(hex(id(list_1)))  # 0x10600fa50
print(hex(id(list_2)))  # 0x1060896e0

list_3 = [item for item in list_2 if item % 2 == 0]
print(list_3)  # [2, 4, 6]
print(hex(id(list_3)))  # 0x106020be0

# Dict comprehension
list_1 = ['a','b','c','d','e','f']
dict_1 = {key: value for key, value in enumerate(list_1)}
# {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}


# Class
class Dog(object):

    def __init__(self, name):
        self.name = name

    def print_name(self):
        print(self.name)


# Object
dog_obj = Dog('Buck')
dog_obj.print_name()


class Dog(object):

    def __new__(cls, name):
        print("__new__")
        return super(Dog, cls).__new__(cls)

    def __init__(self, name):
        print("__init__")
        self.name = name

    def print_name(self):
        print(self.name)


dog_obj_1 = Dog("Buck")
dog_obj_2 = Dog("Carmy")
print(hex(id(dog_obj_1)))  # 0x102ac3090
print(hex(id(dog_obj_2)))  # 0x102ac36d0


class Dog_2(object):

    instance = None

    def __new__(cls, name):
        if Dog_2.instance is None:
            print("__new__ object created")
            Dog_2.instance = super(Dog_2, cls).__new__(cls)
            return Dog_2.instance
        else:
            return Dog_2.instance

    def __init__(self, name):
        print("__init__")
        self.name = name

    def print_name(self):
        print(self.name)


dog_obj = Dog_2('Buck')
dog_obj.print_name()



class Animal(object):

    def __init__(self, name):
        self.name = name

    def print_name(self):
        print("Animal con nombre {}".format(self.name))


class Dog(Animal):

    def __init__(self, name):
        self.name = name

    def print_name(self):
        print("Dog with name {}".format(self.name))


class Cat(Animal):

    def __init__(self, name):
        self.name = name

    def print_name(self):
        print("Cat with name {}".format(self.name))


animals = [Animal('Buck'), Dog('Carmy'), Cat('Jan')]
for animal in animals:
    animal.print_name()

# Animal con nombre Buck
# Dog with name Carmy
# Cat with name Jan


# save object in file
import pickle
dog_1 = Dog('Buck')
with open('dataset.pkl', 'wb') as file:
    pickle.dump(dog_1, file, protocol=pickle.HIGHEST_PROTOCOL)


# load object from file
with open('dataset.pkl', 'rb') as file:
    dog_2 = pickle.load(file)
dog_2.print_name()



import numpy as np

array_1 = np.array([1,2,3,4], dtype=np.int64)
print(array_1)          # [1 2 3 4]
print(type(array_1))    # <class 'numpy.ndarray'>
print(array_1.dtype)    # int64
print(array_1.shape)    # (4,)

array_2 = np.array([[1,2,3],[4,5,6]], dtype=np.float64)
print(array_2)          # [[1. 2. 3.]
                        #  [4. 5. 6.]]
print(type(array_2))    # <class 'numpy.ndarray'>
print(array_2.dtype)    # float64
print(array_2.shape)    # (2, 3)




array_1 = np.array([1,2,3,4], dtype=np.int64)
array_1[::-1]     # array([4, 3, 2, 1])
array_1[0:1]      # array([1])
array_1[0:2]      # array([1, 2])
array_1[:2]       # array([1, 2])
array_1[1:]       # array([2, 3, 4])
array_1[0:3:2]    # array([1, 3])
array_1[[3,1]]    # array([4,2])
array_1[[3,1,0]]  # array([4,2,1])


array_2 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=np.float64)
array_2[0:2, 1:3]      # array([[2., 3.],
                       #        [6., 7.]])
array_2[(1,2),(2,3)]   # array([ 7., 12.])



array_1 = np.array([1,2,3,4,5,6,7,8])
mask_1 = array_1 > 4
print(mask_1)      # [False False False False  True  True  True  True]
array_2 = array_1[mask_1]
print(array_2)     # [5 6 7 8]

array_1 = np.array([1,2,3,4,5,6,7,8])
mask_1 = array_1 > 4
mask_2 = array_1 < 8
print(mask_1)      # [False False False False  True  True  True  True]
print(mask_2)      # [ True  True  True  True  True  True  True False]
array_2 = array_1[mask_1 & mask_2]
print(array_2)     # [5 6 7]



# element wise operations
a = np.array([1,2,3,4])
b = np.array([1,2,3,4])
c = a * b
print(c) # array([ 1,  4,  9, 16])

# broadcasting number to 1d array
a = np.array([1,2,3,4,5,6,7,8])
b = a + 1
print(b)  # array([2, 3, 4, 5, 6, 7, 8, 9])
          # number 1 was broadcasted and added to every element of a

# broadcasting 1d array to 2d array
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b = np.array([10,10,10,10])
c = a + b
print(c)  # array([[11, 12, 13, 14],
          #        [15, 16, 17, 18],
          #        [19, 20, 21, 22]])
          # 1d array [10,10,10,10] was broadcasted and added element wise to every row of c




structure = [('word', np.dtype('U60')),
             ('embedding', np.float32, (4,))]
structure = np.dtype(structure)
a = np.empty(10, dtype=structure)
a[0]['word'] = 'Hola'
a[0]['embedding'] = [1,2,3,4]
a[1]['word'] = 'Chau'
a[1]['embedding'] = [-1,-2,-3,-4]

# array([
#     ('Hola', [ 1.,  2.,  3.,  4.]),
#     ('Chau', [-1., -2., -3., -4.]),
#     ('', [ 0.,  0.,  0.,  0.]),
#     ('', [ 0.,  0.,  0.,  0.]),
#     ('', [ 0.,  0.,  0.,  0.]),
#     ('', [ 0.,  0.,  0.,  0.]),
#     ('', [ 0.,  0.,  0.,  0.]),
#     ('', [ 0.,  0.,  0.,  0.]),
#     ('', [ 0.,  0.,  0.,  0.]),
#     ('', [ 0.,  0.,  0.,  0.])],
# dtype=[('word', '<U60'), ('embedding', '<f4', (4,))])
