import pandas as pd
import numpy as np
import sys, getopt


def perceptron(weights_vector, X):
    product_of_weights_inputs = X.dot(weights_vector)
    output = product_of_weights_inputs.apply(lambda x: 1 if x > 0 else 0)
    return output


def weight_update(old_weight, learning_rate, y, y_pred, x):
    new_weight = old_weight + sum(learning_rate * ((y - y_pred) * x))
    return new_weight


def init(input_filename, output_filename):
    # def init():
    learning_rate = 1
    dataset = pd.read_csv(input_filename, delimiter='\t', header=None)

    dataset = dataset.dropna(axis=1)

    # dataset[1] = dataset[1].apply(lambda x: round(x, 2))
    # dataset[2] = dataset[2].apply(lambda x: round(x, 2))
    dataset[3] = 1

    dataset[0] = dataset[0].apply(lambda x: 1 if x == 'A' else 0)

    weights_vector_constant_learning = np.zeros(len(dataset.columns) - 1)
    weights_vector_annealing_learning = np.zeros(len(dataset.columns) - 1)

    X = dataset.drop(0, axis=1)
    y = dataset[0]

    error_df = pd.DataFrame(index=range(2))
    for itr in range(101):
        y_pred_constant_learning = perceptron(weights_vector_constant_learning, X)
        error_rate_constant_learning = sum(abs(y - y_pred_constant_learning))

        y_pred_annealing_learning = perceptron(weights_vector_annealing_learning, X)
        error_rate_annealing_learning = sum(abs(y - y_pred_annealing_learning))

        error_df[itr] = [error_rate_constant_learning, error_rate_annealing_learning]

        new_weights_constant_learning = []
        for i, weight in enumerate(weights_vector_constant_learning):
            new_weights_constant_learning.append(
                weight_update(weight, learning_rate, y, y_pred_constant_learning, X[i + 1]))
        weights_vector_constant_learning = new_weights_constant_learning

        new_weights_annealing_learning = []
        for i, weight in enumerate(weights_vector_annealing_learning):
            new_weights_annealing_learning.append(
                weight_update(weight, learning_rate / (itr + 1), y, y_pred_annealing_learning, X[i + 1]))
        weights_vector_annealing_learning = new_weights_annealing_learning

    error_df.to_csv(output_filename, sep='\t', header=False, index=False)


if __name__ == "__main__":
    input_filename = None
    output_filename = None
    # passing arguments for command line execution
    opts = getopt.getopt(sys.argv[1:], '', ["data=", "output="])
    opts = opts[0]
    # print(opts)
    for opt, arg in opts:
        if opt == '--data':
            input_filename = arg
        elif opt == '--output':
            output_filename = arg

    if input_filename == None or output_filename == None:
        print('Please provide all the inputs: input_filename , output_filename')
        exit()
    init(input_filename, output_filename)
    # init()
