# Root Finding Method
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plot_function(func, a, b, roots_bisection):
    """
    This function plots the graph of the input func 
    within the given intervals [a[i],b[i]) along with the roots
    obtained from Bisection Method and Newton’s Method.
    """
    for i in range(len(a)):
        x_val = np.linspace(a[i], b[i], 1000)  # Generate 1000 points between a[i] and b[i]
        y_val = func(x_val)  # Calculate corresponding y values
        lbl = '[' + str(a[i]) + ',' + str(b[i]) + ']'
        plt.plot(x_val, y_val, label=lbl)
        
        # Plot roots obtained from Bisection Method
        plt.plot(roots_bisection[i], func(roots_bisection[i]), 'ro', markersize=8)
    
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Graphs of Functions with Roots')
    plt.grid(True)
    plt.legend()
    plt.show()

    


def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method to find the root of a function within a given interval.

    Parameters:
    - func: The function for which the root is to be found.
    - a, b: Interval [a, b] within which the root is searched for.
    - tol: Tolerance level for checking convergence of the method.
    - max_iter: Maximum number of iterations.

    Returns:
    - root: Approximation of the root.
    
    Example
    --------
    >>> fun = lambda x: x**2 - x - 1
    >>> root = bisection_method(fun, 1, 2, max_iter=20)
    """

    # Check if the interval is valid (signs of f(a) and f(b) are different)
    # Your code goes here
    #if np.sign(func(a)) == np.sign(func(b)):
       # raise ValueError("The signs of f(a) and f(b) must be different.")
    # Main loop starts here
    iter_count = 1
    while iter_count <= max_iter:
        # your code goes here
        c = (a + b) / 2  # midpoint
        if abs(func(c)) < tol or abs(b - a) < tol:
        #if abs(b - a) < tol:
            return c  # Found root 
        iter_count += 1 
        if func(c) * func(a) < 0:
            b = c 
        else:
            a = c
        
    
    print("Warning! Exceeded the maximum number of iterations.")
    return c

# Example usage:
if __name__ == "__main__":
    # Define the function for which the root is to be found
    func = lambda x: x**2 - x - 1  # First Function
    
    # Uncomment the below line to use the Second Function
    #func = lambda x: x**3 - x**2 - 2*x + 1  # Second Function



    # Set the interval [a, b] for the search
    a_1 = 1; b_1 = 2;  # For first root (change the values as required)
    a_2 = -1; b_2 = 0;  # For second root (change the values as required)
    a_3 = 2; b_3 = 3;
    # Call plot_function to plot graph of the function
    # Your code goes here

    #plot_function(func, a_1, b_1)
    #plot_function(func, a_2, b_2)
    #plot_function(func, a_3, b_3)
    # Call the bisection method
    our_root_1 = bisection_method(func, a_1, b_1)
    our_root_2 = bisection_method(func, a_2, b_2)
    #print("a_3 " + str(a_3) + " b_3 " + str(b_3))
    our_root_3 = bisection_method(func, a_3, b_3)

    # Call SciPy method root, which we consider as a reference method.
    x0 = (a_1 + b_1)/2
    sp_result_1 = sp.optimize.root(func, x0)
    sp_root_1 = sp_result_1.x.item()

    x1 = (a_2 + b_2)/2
    sp_result_2 = sp.optimize.root(func, x1)
    sp_root_2 = sp_result_2.x.item()

    x2 = (a_3 + b_3)/2
    sp_result_3 = sp.optimize.root(func, x2)
    sp_root_3 = sp_result_3.x.item()
    a = [a_1, a_2, a_3]
    b = [b_1, b_2, b_3]
    sp_roots  = [sp_root_1, sp_root_2, sp_root_3]
    plot_function(func, a, b, sp_roots)
    # Print the result
    print("1st root found by Bisection Method = {:0.8f}.".format(our_root_1))
    print("1st root found by SciPy = {:0.8f}".format(sp_root_1))

    print("2nd root found by Bisection Method = {:0.8f}.".format(our_root_2))
    print("2nd root found by SciPy = {:0.8f}".format(sp_root_2))

    print("3nd root found by Bisection Method = {:0.8f}.".format(our_root_3))
    print("3nd root found by SciPy = {:0.8f}".format(sp_root_3))
    
