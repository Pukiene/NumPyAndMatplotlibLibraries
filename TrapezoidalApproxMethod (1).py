import numpy as np
import matplotlib.pyplot as plt

def plot_function(function, N1l, trapezoidal_approx):
    if function == 1:
        fText = "Trapezoid Approx Method Relationship (Function 1)"
        fText2 = "Trapezoid Values"
    if function == 2:
        fText = "Trapezoid Approx Method Relationship (Function 2)"
        fText2 = "Trapezoid Values"
    if function == 3:
        fText = "Trapezoid Approx Method Error Relationship (Fuction 1)"
        fText2 = "Trapezoid Error Values"
    if function == 4:
        fText = "Trapezoid Approx Method Error Relationship (Fuction 2)"
        fText2 = "Trapezoid Error Values"

    y_values = np.array(trapezoidal_approx)
    plt.xlabel(fText2)
    plt.ylabel("N Values")    
    
    x_midpoint = np.array(N1l)
    plt.title(fText)
    plt.plot(x_midpoint, y_values)
    plt.show()

def trapezoidal_approx(func, a, b, N):
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

    # Your code goes here
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)
    result =  h * (np.sum(y) - 0.5 * (y[0] + y[-1]))
    return result

if __name__ == "__main__":
    # 1st Function to be integrated
    func_1 = lambda x : x/(x**2 + 1)
    # Indefinite Integral of the function
    antiderivative_1 = lambda x: np.log(x**2 + 1) / 2 # Your code goes here
    
    # 2nd Function to be integrated
    func_2 = lambda x : np.exp(x)
    # Indefinite Integral of the function
    antiderivative_2 = lambda x: np.exp(x)# Your code goes here
    
    # End points for 1st Function
    a1 = 0; b1 = 5;  # Change the values as required
    # End points for 2nd Function
    a2 = 0; b2 = 5;  # Change the values as required
    
    # Number of partition for 1st Function
    N1 = 500 # Change the value as required
    # Number of partition for 2nd Function
    N2 = 500 # Change the value as required
    
    # Call midpont_method to compute Trapezoidal Approximation:
    trapezoidal_approx_1 = trapezoidal_approx(func_1, a1, b1, N1) # Your code for 1st function
    trapezoidal_approx_2 = trapezoidal_approx(func_2, a2, b2, N2) # Your code for 2nd function
    
    # Calculate the true value of the definite integral
    definite_integral_1 = antiderivative_1(b1) - antiderivative_1(a1)  # For 1st Function
    definite_integral_2 = antiderivative_2(b2) - antiderivative_2(a2)  # For 2nd Function

    # Calculate the absolute error between the approximate value and true value
    error_1 = np.abs(trapezoidal_approx_1 - definite_integral_1)  # For 1st Function
    error_2 = np.abs(trapezoidal_approx_2 - definite_integral_2)  # For 2nd Function

    
    print("Trapezoidal Approximation for 1st Function = {:0.6f}".format(trapezoidal_approx_1))
    print("Actual Value for 1st Function = {:0.6f}".format(definite_integral_1))
    print("Absolute error between the above methods = {:0.8f}".format(error_1))

    print("Trapezoidal Approximation for 2nd Function = {:0.6f}".format(trapezoidal_approx_2))
    print("Actual Value for 2nd Function = {:0.6f}".format(definite_integral_2))
    print("Absolute error between the above methods = {:0.8f}".format(error_2))

    #Graphs START
    N1l = [10, 30, 50, 100, 500]

    #definite_integral_1 = float("{:0.8f}".format(definite_integral_1))
    trapezoidal_approx_f1 = [float("{:0.6f}".format(trapezoidal_approx(func_1, a1, b1, N))) for N in N1l]
    trapezoidal_approx_f1e = [float("{:0.12f}".format(np.abs(value - definite_integral_1))) for value in trapezoidal_approx_f1]
    trapezoidal_approx_f2 = [float("{:0.6f}".format(trapezoidal_approx(func_2, a1, b1, N))) for N in N1l]
    trapezoidal_approx_f2e = [float("{:0.12f}".format(np.abs(value - definite_integral_2))) for value in trapezoidal_approx_f2]
    plot_function(1,trapezoidal_approx_f1, N1l)
    plot_function(3,trapezoidal_approx_f1e, N1l)
    plot_function(2,trapezoidal_approx_f2, N1l)
    plot_function(4,trapezoidal_approx_f2e, N1l)
    print(trapezoidal_approx_f1e)
    print(trapezoidal_approx_f1)
    #Graphs END
    