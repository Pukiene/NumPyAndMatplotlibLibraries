import numpy as np
import matplotlib.pyplot as plt

def plot_function(function, N1l, trapezoidal_approx):
    if function == 1:
        fText = "Midpoint Approx Method Relationship (Function 1)"
        fText2 = "Midpoint Values"
    if function == 2:
        fText = "Midpoint Approx Method Relationship (Function 2)"
        fText2 = "Midpoint Values"
    if function == 3:
        fText = "Midpoint Approx Method Error Relationship (Fuction 1)"
        fText2 = "Midpoint Error Values"
    if function == 4:
        fText = "Midpoint Approx Method Error Relationship (Fuction 2)"
        fText2 = "Midpoint Error Values"

    y_values = np.array(trapezoidal_approx)
    plt.xlabel(fText2)
    plt.ylabel("N Values")    
    
    x_midpoint = np.array(N1l)
    plt.title(fText)
    plt.plot(x_midpoint, y_values)
    plt.show()
    

def midpoint_approx(func, a, b, N):
    '''Compute the Midpoint Approximation of Definite Integral of a function over the interval [a,b].

    Parameters
    ----------
    func : function
           Vectorized function of one variable
    a , b : numbers
        Endpoints of the interval [a,b]
    N : integer
        Number of subintervals of equal length in the partition of [a,b]

    Returns
    -------
    float
        Approximation of the definite integral by Midpoint Approximation.
    '''
    h = (b - a) / N
    x = np.linspace(a + h / 2, b - h / 2, N)
    result = np.sum(func(x)) * h
    # Your code goes here
    return result

if __name__ == "__main__":
    # 1st Function to be integrated
    func_1 = lambda x : x/(x**2 + 1)
    # Indefinite Integral of the function
    antiderivative_1 = lambda x: np.log(x**2 + 1) / 2 # Your code goes here
    
    # 2nd Function to be integrated
    func_2 = lambda x : np.exp(x)
    # Indefinite Integral of the function
    antiderivative_2 = lambda x: np.exp(x) # Your code goes here
    
    # End points for 1st Function
    a1 = 0; b1 = 5;  # Change the values as required
    # End points for 2nd Function
    a2 = 0; b2 = 5;  # Change the values as required

    # Call the function to Plot the graph of the functions
    # Your code goes here
    
    #plot_function(func_1, a1, b1)
    #plot_function(func_2, a2, b2)
    # Number of partition for 1st Function
    N1 = 500 # Change the value as required
    # Number of partition for 2nd Function
    N2 = 500 # Change the value as required

    # Call midpont_method to compute Midpoint Approximation:
    midpoint_approx_1 = midpoint_approx(func_1, a1, b1, N1) # Your code for 1st function
    midpoint_approx_2 = midpoint_approx(func_2, a2, b2, N2) # Your code for 2nd function
    
    # Calculate the true value of the definite integral
    definite_integral_1 = antiderivative_1(b1) - antiderivative_1(a1)  # For 1st Function
    definite_integral_2 = antiderivative_2(b2) - antiderivative_2(a2)  # For 2nd Function

    # Calculate the absolute error between the approximate value and true value
    error_1 = np.abs(midpoint_approx_1 - definite_integral_1)  # For 1st Function
    error_2 = np.abs(midpoint_approx_2 - definite_integral_2)  # For 2nd Function

    
    print("Midpoint Approximation for 1st Function = {:0.6f}".format(midpoint_approx_1))
    print("Actual Value for 1st Function = {:0.6f}".format(definite_integral_1))
    print("Absolute error between the above methods = {:0.8f}".format(error_1))

    print("Midpoint Approximation for 2nd Function = {:0.6f}".format(midpoint_approx_2))
    print("Actual Value for 2nd Function = {:0.6f}".format(definite_integral_2))
    print("Absolute error between the above methods = {:0.8f}".format(error_2))


    #Graphs START
    N1l = [10, 30, 50, 100, 500]

    midpoint_approx_f1 = [float("{:0.6f}".format(midpoint_approx(func_1, a1, b1, N))) for N in N1l]
    midpoint_approx_f1e = [float("{:0.12f}".format(np.abs(value - definite_integral_1))) for value in midpoint_approx_f1]
    midpoint_approx_f2 = [float("{:0.6f}".format(midpoint_approx(func_2, a1, b1, N))) for N in N1l]
    midpoint_approx_f2e = [float("{:0.12f}".format(np.abs(value - definite_integral_2))) for value in midpoint_approx_f2]
    plot_function(1,midpoint_approx_f1, N1l)
    plot_function(3,midpoint_approx_f1e, N1l)
    plot_function(2,midpoint_approx_f2, N1l)
    plot_function(4,midpoint_approx_f2e, N1l)
    print(midpoint_approx_f1e)
    print(midpoint_approx_f1)
    #Graphs END