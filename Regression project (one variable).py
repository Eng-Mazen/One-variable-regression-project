try:
    import pandas as pd
    import matplotlib.pyplot as plt 
    import numpy as np
    path = input("Enter the data file full path using (Ctrl+Shift+C) : ").strip()
    x_axis_name = input("Enter the name of x-axis:")
    y_axis_name = input("Enter the name of y-axis:")
    Regression_data = pd.read_excel(rf"{path[1 : len(path)-1]}")
    data = pd.DataFrame(Regression_data)
    X, y = data.iloc[:, 0], data.iloc[:, 1] # iloc is needed
    X = np.array(X).reshape(X.size, 1)
    y = np.array(y).reshape(y.size, 1)
    X0 = np.ones(X.size).reshape(X.size,1)
    theta = np.zeros(2).reshape(2,1)
    # Calculate the cost now
    X = np.hstack((X0, X))
    def calculate_cost(X, y, theta):
        m = len(X)
        formula_before_square = np.dot(X,theta) - y
        cost = (1 / (2 * m)) * np.sum(formula_before_square)**2
        return cost
    init_cost = calculate_cost(X, y, theta)
    print("The initial cost =", init_cost)
    def gradient_descent(X, y, theta, iterations=int(input("Enter the number of iterations :")), alpha=float(input("Enter the value of alpha :"))): # ***  THIS SYNTAX IS VERY IMPORTANT ***
        tries = np.zeros(iterations) # important for history function to find the perfect number of iterations
        m =len(y) # Number of rows
        for i in range(iterations):
            error_value = X.dot(theta) - y
            theta -= (alpha / m) * (X.T.dot(error_value))
            tries[i] = calculate_cost(X, y, theta)
        return theta, tries, iterations
    print("intial cost = ", init_cost)
    X0, X1 = X[:, 0], X[:, 1]
    print(X0)
    print(X1)
    new_theta, trash, iterations = gradient_descent(X, y, theta)
    print("new cost =", calculate_cost(X, y, new_theta))
    def reg_line(x):
        return new_theta[0,0]+ x* new_theta[1,0]
    start_x = np.min(X1)
    end_x = np.max(X1)
    x = np.linspace(start_x, end_x, 1000)
    plt.scatter(X1.T, y, color="b", label="The real data ")
    plt.plot(x, reg_line(x),color="r", label="Prediction line")
    plt.xlabel(f"{x_axis_name}")
    plt.ylabel(f"{y_axis_name}")
    plt.legend()
    plt.show()
    plt.plot(range(iterations), trash)
    plt.xlabel("iteration axis")
    plt.ylabel("Cost value after every single iteration")
    plt.show()

except ValueError :
    print("""You have entered a non-numbers data
          ,Please try again after checking the data""")
except RuntimeWarning :
    print("The value you have entered makes cost become nearly to +inf \n Change it")