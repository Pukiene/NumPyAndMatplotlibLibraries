# Root Finding Method
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plot_function(func, x0, roots_newton):
    """
    This function plots the graph of the input func 
    within the given intervals [a[i],b[i]) along with the roots
    obtained from Bisection Method and Newton’s Method.
    """
    for i in range(len(x0)):
        x_val = np.linspace(x0[i] - 1, x0[i] + 1, 1000)  # Generate 1000 points around x0[i]
        y_val = func(x_val)  # Calculate corresponding y values
        lbl = 'x0=' + str(x0[i])
        plt.plot(x_val, y_val, label=lbl)
        
        # Plot roots obtained from Newton's Method
        plt.plot(roots_newton[i], func(roots_newton[i]), 'go', markersize=8, label="Newton")

    plt.plot(x_val, y_val)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of the function')
    plt.grid(True)
    plt.show()
    
def newton_method(func, grad, x0, tol=1e-6, max_iter=100):
    '''Approximate solution of f(x)=0 by Newton-Raphson's method.

        Parameters
        ----------
        func : function 
            Function value for which we are searching for a solution f(x)=0,
        grad: function
            Gradient value of function f(x)
        x0 : number
            Initial guess for a solution f(x)=0.
        tol : number
            Stopping criteria is abs(f(x)) < tol.
        max_iter : integer
            Maximum number of iterations of Newton's method.

        Returns
        -------
        xn : root

        Example
        --------
        >>> fun = lambda x: x**2 - x - 1
        >>> grad = lambda x: 2*x - 1
        >>> root = newton_method(fun, grad, 1, max_iter=20)
        '''
    # Main Loop starts here
    iter_count = 1
    while iter_count <= max_iter:
        xn = x0 - func(x0) / grad(x0)  # Newton's method iteration
        if abs(func(xn)) < tol:            
            return xn  # Found the root
        x0 = xn  # Update the guess for the next iteration
        root = xn 
        iter_count += 1
        

    print("Warning! Exceeded the maximum number of iterations.")
    return root


# Main Driver Function:
if __name__ == "__main__":
    # Define the 1st Function for which the root is to be found
    func = lambda x: x**2 - x - 1    
    grad = lambda x: 2*x -1

    # Uncomment the next two lines to use the 2nd Function
    #func = lambda x: x**3 - x**2 - 2*x + 1
    #grad = lambda x: 3*x**2 - 2*x -2

    # Call plot_function to plot graph of the function
    # Your code goes here
   

    x0 = 0 # Initial guess for 1st (change the value as required)
    # Call the Newton's method for 1st root
    our_root_1 = newton_method(func, grad, x0) # Your code goes here

    # Call SciPy method (reference method) for 1st root
    sp_result_1 = sp.optimize.root(func, x0)
    sp_root_1 = sp_result_1.x.item()

    # Call the Newton's method for 1nd root
    x0 = 1 # Initial guess for 2nd root (change the value as required)
    our_root_2 = newton_method(func, grad, x0) # Your code goes here

    # Call SciPy method (reference method) for 2nd root
    sp_result_2 = sp.optimize.root(func, x0)
    sp_root_2 = sp_result_2.x.item()

    x0 = -1 # Initial guess for 3nd root (change the value as required)
    our_root_3 = newton_method(func, grad, x0) # Your code goes here

    # Call SciPy method (reference method) for 3nd root
    sp_result_3 = sp.optimize.root(func, x0)
    sp_root_3 = sp_result_3.x.item()
    x0s = [0, 1, -1]
    sp_roots  = [sp_root_1, sp_root_2, sp_root_3]
    plot_function(func, x0s, sp_roots)
    # Print the result
    print("1st root found by Newton's Method = {:0.8f}.".format(our_root_1))
    print("1st root found by SciPy = {:0.8f}".format(sp_root_1))

    print("2nd root found by Newton's Method = {:0.8f}.".format(our_root_2))
    print("2nd root found by SciPy = {:0.8f}".format(sp_root_2))
    
    print("3nd root found by Newton's Method = {:0.8f}.".format(our_root_3))
    print("3nd root found by SciPy = {:0.8f}".format(sp_root_3))
    
