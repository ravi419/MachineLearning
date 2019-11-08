import pandas as pd
import numpy as np
import sys, getopt

def init(input_filename, output_filename):
# def init():
    dataset = pd.read_csv(input_filename, delimiter='\t', header=None)
    dataset= dataset.dropna(axis=1, how='all')
    dataset = dataset.rename({0:'label'}, axis=1)
    attributes = dataset.drop('label', axis=1).columns.tolist()
    classes = dataset['label'].unique().tolist()

    class_probabilities = (dataset['label'].value_counts() / dataset.shape[0]).to_dict()
    att_mean_dict = dataset.groupby('label').mean().to_dict()
    att_var_dict = dataset.groupby('label').var().to_dict()

    for ind in dataset.index:
        sample = dataset.loc[ind].to_dict()

        post_1_A = (1 / np.sqrt(2 * np.pi * att_var_dict[1]['A'])) * np.exp(
            -(sample[1] - att_mean_dict[1]['A']) ** 2 / (2 * att_var_dict[1]['A']))

        post_1_B = (1 / np.sqrt(2 * np.pi * att_var_dict[1]['B'])) * np.exp(
            -(sample[1] - att_mean_dict[1]['B']) ** 2 / (2 * att_var_dict[1]['B']))

        post_2_A = (1 / np.sqrt(2 * np.pi * att_var_dict[2]['A'])) * np.exp(
            -(sample[2] - att_mean_dict[2]['A']) ** 2 / (2 * att_var_dict[2]['A']))

        post_2_B = (1 / np.sqrt(2 * np.pi * att_var_dict[2]['B'])) * np.exp(
            -(sample[2] - att_mean_dict[2]['B']) ** 2 / (2 * att_var_dict[2]['B']))

        prob_A = (post_1_A * post_2_A * class_probabilities['A']) / (
                    (post_1_A * post_2_A * class_probabilities['A']) + (post_1_B * post_2_B * class_probabilities['B']))

        prob_B = (post_1_B * post_2_B * class_probabilities['B']) / (
                    (post_1_A * post_2_A * class_probabilities['A']) + (post_1_B * post_2_B * class_probabilities['B']))

        if prob_A >= prob_B:
            dataset.at[ind, 'pred'] = 'A'
        else:
            dataset.at[ind, 'pred'] = 'B'

    dataset['miss'] = dataset['label'] != dataset['pred']
    dataset['miss'] = dataset['miss'].apply(lambda x: 1 if x==True else 0)
    # print('Number of misclassifications: ', dataset['miss'].sum())

    with open(output_filename, 'w') as file:
        for class_label in classes:
            for column in attributes:
                file.write(str(att_mean_dict[column][class_label]))
                file.write('\t')
                file.write(str(att_var_dict[column][class_label]))
                file.write('\t')
            file.write(str(class_probabilities[class_label]))
            file.write('\n')
        file.write(str(dataset['miss'].sum()))

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