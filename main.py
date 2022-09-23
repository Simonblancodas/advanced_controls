# The following Script is meant to develop a function for the Realization of State Space
# which means that it will take the Transfer Function of a controllable system (with certain limitations) and allow the
# user to obtain the State Space equation in one of three optional forms: Controllable, Observable or Diagonal

# We start by importing all the "libraries" that we need for our program.

import numpy as np  # the fundamental package for scientific computing in Python
import spicy as sp  # the fundamental package for control systems operations in Python

numerator = [2, 4, 6, 8]
denominator = [135, 5, 65, 9, 1]

# Now we define our function with it's corresponding documentation.

def TF2SS(numerator, denominator, option):

    '''
    Here is a short explanation of how the function works:
    TF2SS means Transfer Function to State Space,

    Assumptions/ restrictions:
    -- The numerator and denominator of the TF must be polynomial equations
    -- The degree of the equation in the numerator must be lower than that of the denominator
    -- The SS equation is not unique, hence the form must be specified
    -- The system must have distinct and real poles

    Inputs:
    numerator: corresponds to constants accompanying the variable of the polynomial equation.
                ++ the input must be in the following format: [b0, b1, b2, ..., bn]
                -- where b0 corresponds to the constant with no multiplying variable and the bn to the one multiplying
                   the variable with the highest exponent.

    denominator: corresponds to constants accompanying the variables of the polynomial equation, except the term of
                 the highest order.
                ++ the input must be in the following format: [a0, a1, a2, ..., an]
                -- where a0 corresponds to the constant with no multiplying variable and the bn to the one multiplying
                   the variable with the highest exponent.

    option: correspond to the type of canonical form of the State Space Equation that we want as an output
                ++ The input must be in the following format: 'c', 'o', or 'd'
                -- where 'c' will return the Controllable Canonical form, 'o' will return the Observable Canonical
                   form, and 'd' will return the Canonical Diagonal form.

    Outputs:
    A: will be the nxn matrix that multiplies the X(t) vector of x dot
    B: will be the nx1 vector that multiplies the U(t) vector of x dot
    C: will be the 1xn vector that multiplies the X(t) vector of y
    D: will be the constant that multiplies the U(t) vector of y

    '''

    np.set_printoptions(precision=3)  # this command makes python show only 3 decimal points

    # Controllable Form

    # we count the number of terms of the inputs to determine the degree of the polynomials
    num_degree = len(numerator)
    den_degree = len(denominator)

    n = den_degree-1

    # now we compare the degrees to guarantee that our function is capable of finding a solution
    if num_degree >= den_degree:
        print('The degree of the numerator must be lower than the denominator')

    # we transform our inputs into manipulable objects
    num_arr = np.array(numerator)
    den_arr = np.array(denominator)

    # here we guarantee that the term accompanying the variable of the highest order in the denominator is equal to 1
    last_term = denominator[len(denominator)-1]
    if last_term != 1:
        den_arr = den_arr/last_term
        num_arr = num_arr/last_term

    # we create our output variables, originally empty

    A = []
    B = []
    C = []
    D = []
    Z = []
    ss = []

    den_arr = np.delete(den_arr, n)  # here we drop the last term of the input which is equal to one

    I = np.identity(n-1)  # we create an identity matrix that will be (n-1)x(n-1)
    for i in range(n-1):  # we create a vector of zeroes that will be (n-1)x1
        Z.append([0])
    Z_arr = np.array(Z)  # the vector was transformed into a form that can be manipulated

    #  Controllable form

    if option == 'c':
        A = np.hstack([Z_arr, I])  # we stack horizontally the Zero vector and Identity matrix
        neg_den = den_arr*-1  # we multiply the denominator constants by negative one (-1)
        A = np.vstack([A, neg_den])  # we finish stacking the matrix A with the negative constants to get nxn matrix
        B = np.vstack([Z_arr, 1])  # we add a one to the zero vertical matrix to create B to have a nx1 vector
        C = num_arr.T  # we create Z by simply transposing the constants of the numerator to have a 1xn vector
        D = [[0]]  # because of our assumptions, D is always equal to zero

        ss = [A, B, C, D]

    #   Observable form

    if option == 'o':
        A = np.vstack([I, Z_arr.T])  # we stack the Identity matrix with a vector of zeroes
        rev_den = np.flip(den_arr)  # we flip the denominator to get the correct order for A
        den_T = []  # we create a vector of the correct form with the values of the last row of A
        for i in range(n):
            den_T.append([rev_den[i]])
        den_T_arr = np.array(den_T)
        neg_den_T = den_T_arr*-1
        A = np.hstack([neg_den_T, A])  # we get A by stacking the constants of the denominator with the previoew stack
        B_arr = np.flip(num_arr)
        for i in range(n):
            B.append([num_arr[i]])  # we flip the numerator array and transform it to a nx1 vector
        B = np.array(B)
        C = np.vstack([1, Z_arr]).T  # we add a one to the zero vector at the beginning and transpose to get 1xn vector
        D = [[0]]  # because of our assumptions, D is always equal to zero

        ss = [A, B, C, D]

    #   Diagonal form

    if option == 'd':
        [r, p, k] = sp.signal.residue(numerator, denominator)  # we use the residue function to find
                                                                # the Residues corresponding to the poles (r),
                                                                # Poles ordered by magnitude in ascending order (p),
                                                                # and the coefficients of the direct polynomial term

        A = np.diag(p)
        for i in range(n):
            B.append([1])
        B = np.array(B)
        C = r
        D = [[k]]

        ss = [A, B, C, D]

    return ss  # the function returns a list of the four (4) terms of the State Space equations








