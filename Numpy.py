import numpy as np

arr = np.array([1,2,3]) ## it converts the array to matrix
my_mat = [[1,23,],[4,5,6],[7,8,9]]
mat = np.array(my_mat)

print(mat)

ar = np.arange(0,10)
ar = np.arange(0,10,2)
print(ar)

z1 = np.zeros(4) ## we can use ones also
z2 = np.zeros((3,5))
print(z1,end='\n\n')
print(z2)

# lin = np.linspace(0,5,10)  # even spaces between the 10 values between 0 and 5

# np.eye(5)*8 # an I matrix

'''
    Random numbers
'''
# print(np.random.rand(6)) # random array in range of 0-1 with length of 6
# print(np.random.rand(6,3)) # random matrix in range of 0-1 with 6X3

# print(np.random.randn(3)) ## returns 3 numbers from the gauss distribution
# print(np.random.randn(3,2)) ## returns 3 numbers from the gauss distribution

# np.random.randint(1,133,4) # normal random


'''
    Using arrays
'''

# arr = np.arange(25)
# rearr = arr.reshape(5,5) ## reshaping the array to matrix


# We can use the .max() .min() methods in those arrays
# ranarr = np.random.randint(0,20,6)
# print(ranarr)
# print(ranarr.argmax()) ## prints the index of the max value--> argmin()

## using the .shape() method gives you the shape of the array as tuple

# print(arr.dtype)   # gives you the data types in the array

'''
    Some Tricks
'''
# arr = np.arange(0,11)
# arr[0:4]=15
# print(arr)

# arr_copty = arr[:5].copy() # deep copy of the array not pointer
# print(arr_copty)

# arr = np.array([[1,2,3],[5,10,15],[20,25,30]]) # using the [1,2] instead of [1][2]
# arr[:2,1:]  # crops the first two lines and from the second column to the end

# We can create a boolean array
# bool_arr = arr<5
# print(bool_arr)
# chose_arr = arr[bool_arr] # => arr[arr<5]
# print(chose_arr)  # creates an array of the only true values -> creates a copy of the roriginal array

'''
    Operations on arrays and universal functions
    https://numpy.org/doc/stable/reference/ufuncs.html or in the notebook
'''
arr = np.arange(0,10)
arr = arr.reshape(2,5)

# print(arr+arr)
# print(arr+100)
# print(arr-arr)
# print(arr*2)
# print(arr*arr)

# print(np.sqrt(arr))
# print(np.exp(arr))
# print(np.sin(arr))
# print(np.cos(arr))
# print(np.log(arr))

# print(arr.sum())


# mat = np.arange(1,26).reshape(5,5)

# print(mat)
# print()
# print(mat.sum(axis=0)) ## --> axis = 0 means sum the matrix as columns, axis = 1 means sum the matrix as rows
