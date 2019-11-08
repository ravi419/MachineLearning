import pandas as pd
import numpy as np
import sys, getopt


def test(case_base, dataset, k):
    missclassification = 0
    for i, sample in enumerate(dataset.values):
        distances_df = case_base.drop('label', axis=1).apply(
            lambda x: np.sqrt((x[1] - sample[1]) ** 2 + (x[2] - sample[2]) ** 2), axis=1)

        distances_df = pd.DataFrame(distances_df)
        distances_df['label'] = case_base['label']
        distances_df = distances_df.sort_values(0, ascending=True)
        distances_df = distances_df.iloc[0:k]

        nn = distances_df.iloc[0].values[0]
        max = distances_df.iloc[-1].values[0]

        weight_df = pd.DataFrame(distances_df[0].apply(lambda x: (max - x) / (max - nn) if max != nn else 1))
        weight_df['label'] = distances_df['label']

        if weight_df.groupby('label').sum().shape[0] == 1:
            label = weight_df.groupby('label').sum().index[0]

        elif weight_df.groupby('label').sum().loc['A'].values[0] > weight_df.groupby('label').sum().loc['B'].values[0]:
            label = 'A'
        else:
            label = 'B'

        if label != sample[0]:
            missclassification = missclassification + 1

    return missclassification

def init(input_filename, output_filename):
# def init():
    dataset = pd.read_csv(input_filename, delimiter='\t', header=None)
    dataset= dataset.dropna(axis=1, how='all')
    dataset = dataset.rename({0:'label'}, axis=1)
    attributes = dataset.drop('label', axis=1).columns.tolist()
    classes = dataset['label'].unique().tolist()
    missclassification_list = []
    for k in [2, 4, 6, 8, 10]:
        test_df = dataset
        case_base = pd.DataFrame()
        case_base = case_base.append(dataset.loc[0])
        for i, sample in enumerate(dataset.values):
            if i == 0:
                continue
            distances_df = case_base.drop('label', axis=1).apply(
                lambda x: np.sqrt((x[1] - sample[1]) ** 2 + (x[2] - sample[2]) ** 2), axis=1)

            distances_df = pd.DataFrame(distances_df)
            distances_df['label'] = case_base['label']
            distances_df = distances_df.sort_values(0, ascending=True)
            distances_df = distances_df.iloc[0:k]

            nn = distances_df.iloc[0].values[0]
            max = distances_df.iloc[-1].values[0]

            weight_df = pd.DataFrame(distances_df[0].apply(lambda x: (max - x) / (max - nn) if max != nn else 1))
            weight_df['label'] = distances_df['label']

            if weight_df.groupby('label').sum().shape[0] == 1:
                label = weight_df.groupby('label').sum().index[0]

            elif weight_df.groupby('label').sum().loc['A'].values[0] > weight_df.groupby('label').sum().loc['B'].values[0]:
                label = 'A'
            else:
                label = 'B'

            if label != sample[0]:
                case_base = case_base.append({'label': sample[0], 1: sample[1], 2: sample[2]}, ignore_index=True)
                test_df = test_df.drop(i)

        if k==4:
            nn_4_cb = case_base
        missclassification_list.append(test(case_base, test_df, k))





    with open(output_filename, 'w') as file:
        for j, miss in enumerate(missclassification_list):
            file.write(str(miss))
            if j != len(missclassification_list)-1:
                file.write('\t')
        file.write('\n')

        for a, b, c in nn_4_cb.values:
            file.write(str(c))
            file.write('\t')
            file.write(str(round(a, 6)))
            file.write('\t')
            file.write(str(round(b, 6)))
            file.write('\n')

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