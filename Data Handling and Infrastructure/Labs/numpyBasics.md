NumPy

NumPy is a popular library for storing arrays of numbers and performing computations on them. Not only this enables to write often clearer code, this also makes the code faster, since most NumPy routines are implemented in C for speed.

To download this notebook click on this link

To install numpy in your environemt use the following command:

!pip install numpy

To use NumPy in your program, you need to import it as follows

import numpy as np

Array creation

NumPy arrays can be created from Python lists

my_array = np.array([1, 2, 3])
my_array

NumPy supports array of arbitrary dimension. For example, we can create two-dimensional arrays (e.g. to store a matrix) as follows

my_2d_array = np.array([[1, 2, 3], [4, 5, 6]])
my_2d_array

We can access individual elements of a 2d-array using two indices

my_2d_array[1, 2]

We can also access rows

my_2d_array[1]

and columns

my_2d_array[:, 2]

Arrays have a shape attribute

print(my_array.shape)
print(my_2d_array.shape)

Contrary to Python lists, NumPy arrays must have a type and all elements of the array must have the same type.

my_array.dtype

The main types are int32 (32-bit integers), int64 (64-bit integers), float32 (32-bit real values) and float64 (64-bit real values).

The dtype can be specified when creating the array

my_array = np.array([1, 2, 3], dtype=np.float64)
my_array.dtype

We can create arrays of all zeros using

zero_array = np.zeros((2, 3))
zero_array

and similarly for all ones using ones instead of zeros.

We can create a range of values using

np.arange(5)

or specifying the starting point

np.arange(3, 5)

Another useful routine is linspace for creating linearly spaced values in an interval. For instance, to create 10 values in [0, 1], we can use

np.linspace(0, 1, 10)

Another important operation is reshape, for changing the shape of an array

my_array = np.array([1, 2, 3, 4, 5, 6])
my_array.reshape(3, 2)

Play with these operations and make sure you understand them well.
Basic operations

In NumPy, we express computations directly over arrays. This makes the code much more succint.

Arithmetic operations can be performed directly over arrays. For instance, assuming two arrays have a compatible shape, we can add them as follows

array_a = np.array([1, 2, 3])
array_b = np.array([4, 5, 6])
array_a + array_b

Compare this with the equivalent computation using a for loop

array_out = np.zeros_like(array_a)
for i in range(len(array_a)):
  array_out[i] = array_a[i] + array_b[i]
array_out

Not only this code is more verbose, it will also run much more slowly.

In NumPy, functions that operates on arrays in an element-wise fashion are called universal functions. For instance, this is the case of np.sin

np.sin(array_a)

Vector inner product can be performed using np.dot

np.dot(array_a, array_b)

When the two arguments to np.dot are both 2d arrays, np.dot becomes matrix multiplication

array_A = np.random.rand(5, 3)
array_B = np.random.randn(3, 4)
np.dot(array_A, array_B)

Matrix transpose can be done using .transpose() or .T for short

array_A.T

Slicing and masking

Like Python lists, NumPy arrays support slicing

np.arange(10)[5:]

We can also select only certain elements from the array

x = np.arange(10)
mask = x >= 5
x[mask]

Exercises

Exercise 1. Create a 3d array of shape (2, 2, 2), containing 8 values. Access individual elements and slices.


Exercise 2. Rewrite the relu function (see Python section) using np.maximum. Check that it works on both a single value and on an array of values.

def relu_numpy(x):
  return

relu_numpy(np.array([1, -3, 2.5]))

Exercise 3. Rewrite the Euclidean norm of a vector (1d array) using NumPy (without for loop)

def euclidean_norm_numpy(x):
  return

my_vector = np.array([0.5, -1.2, 3.3, 4.5])
euclidean_norm_numpy(my_vector)

Exercise 4. Write a function that computes the Euclidean norms of a matrix (2d array) in a row-wise fashion. Hint: use the axis argument of np.sum.

def euclidean_norm_2d(X):
  return

my_matrix = np.array([[0.5, -1.2, 4.5],
                      [-3.2, 1.9, 2.7]])
# Should return an array of size 2.
euclidean_norm_2d(my_matrix)

Exercise 5. Compute the mean value of the features in the iris dataset. Hint: use the axis argument on np.mean.

!pip install sklearn

from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# Result should be an array of size 4.

Matplotlib
Basic plots

Matplotlib is a plotting library for Python.

We start with a rudimentary plotting example.

from matplotlib import pyplot as plt

x_values = np.linspace(-3, 3, 100)

plt.figure()
plt.plot(x_values, np.sin(x_values), label="Sinusoid")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Matplotlib example")
plt.legend(loc="upper left")
plt.show()

We continue with a rudimentary scatter plot example. This example displays samples from the iris dataset using the first two features. Colors indicate class membership (there are 3 classes).

from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

X_class0 = X[y == 0]
X_class1 = X[y == 1]
X_class2 = X[y == 2]

plt.figure()
plt.scatter(X_class0[:, 0], X_class0[:, 1], label="Class 0", color="C0")
plt.scatter(X_class1[:, 0], X_class1[:, 1], label="Class 1", color="C1")
plt.scatter(X_class2[:, 0], X_class2[:, 1], label="Class 2", color="C2")
plt.show()

We see that samples belonging to class 0 can be linearly separated from the rest using only the first two features.
Exercises

Exercise 1. Plot the relu and the softplus functions on the same graph.


What is the main difference between the two functions?

Exercise 2. Repeat the same scatter plot but using the digits dataset instead.

from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)

Are pixel values good features for classifying samples?
