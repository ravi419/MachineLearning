
# libraries
import numpy as np
import pandas as pd
import sys, getopt

# Gradient calculation
def calculate_gradient_for_current_weight(dataset, weight_index):
    gradient_df = (dataset['y_true'] - dataset['y_pred']) * dataset[weight_index]
    gradient_value = gradient_df.sum()
    return gradient_value

def error_difference(new_weights, previous_error, diff_of_errors, threshold, iters, dataset,
                         learning_rate, initial_weights, file):

    while abs(diff_of_errors) > threshold:
        if iters != 0:
            previous_error = sum_of_squared_errors
        dataset['y_pred'] = dataset.drop(['y_true', 'y_pred'], axis=1).dot(initial_weights)
        sum_of_squared_errors = (((dataset['y_true'] - dataset['y_pred']) ** 2).sum())

        # writing according to the given order (iter ,w0,w1,w2....,sum_of_squared_errors)
        file.write(str(iters))
        file.write(',')
        file.write(str("{0:.4f}".format(round(initial_weights[-1], 4))))
        file.write(',')
        for index in range(len(initial_weights) - 1):
            file.write(str("{0:.4f}".format(round(initial_weights[index], 4))))
            file.write(',')
        file.write("{0:.4f}".format(round(sum_of_squared_errors, 4)))
        file.write('\n')

        for weight_index in range(len(initial_weights)):
            # Calculating  Gradient
            gradient_value = calculate_gradient_for_current_weight(dataset, weight_index)
            # updating the weights and appending
            updated_weight = initial_weights[weight_index] + (learning_rate * gradient_value)
            new_weights.append(updated_weight)

        initial_weights = new_weights
        new_weights = []
        iters = iters + 1
        diff_of_errors = sum_of_squared_errors - previous_error

def init(input_filename, learningRate, threshold):
    # reading dataset
    dataset = pd.read_csv(input_filename, delimiter=',', header=None)
    learning_rate = learningRate
    threshold = threshold
    iters = 0
    initial_weights = np.zeros(len(dataset.columns))
    new_weights = []

    dataset = dataset.rename({(len(dataset.columns)-1):'y_true'}, axis=1)
    # Adding Bias as 1 at the end of the dataset
    dataset[len(dataset.columns)-1] = 1
    # Drop y_true and perform dot function to the rest of the dataset
    dataset['y_pred'] = dataset.drop(['y_true'], axis=1).dot(initial_weights)
    # Calculating 1st sum of squared errors
    sum_of_squared_errors = (((dataset['y_true'] - dataset['y_pred']) ** 2).sum())

    # Calculating difference
    previous_error = 0
    diff_of_errors = sum_of_squared_errors - previous_error

    if (input_filename == 'random.csv'):
        file = open('random_solution.csv' , 'w')
        error_difference(new_weights, previous_error, diff_of_errors, threshold, iters, dataset,
                         learning_rate, initial_weights, file)
    else:
        file = open('yacht_solutions.csv' , 'w')
        error_difference(new_weights, previous_error, diff_of_errors, threshold, iters, dataset,
                         learning_rate, initial_weights, file)

if __name__ == "__main__":
    input_filename = None
    learningRate = None
    threshold = None
    # passing arguments for command line execution
    opts = getopt.getopt(sys.argv[1:], '', ["data=", "learningRate=", "threshold="])
    opts = opts[0]
    print(opts)
    for opt, arg in opts:
        if opt == '--data':
            input_filename = arg
        elif opt == '--learningRate':
            learningRate = float(arg)
        elif opt == '--threshold':
            threshold = float(arg)

    if input_filename == None or learningRate == None or threshold == None:
        print('Please provide all the inputs: input_filename, learningRate, threshold')
        exit()
    init(input_filename, learningRate, threshold)
