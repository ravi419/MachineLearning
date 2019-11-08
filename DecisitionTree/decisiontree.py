# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:34:29 2018

@author: Ravi
"""
import numpy as np
import pandas as pd
import xml.etree.cElementTree as ET
import sys, getopt


#Calculating gain
def gain(ent_tar, attr_ent):
    g = ent_tar - attr_ent
    return g

#Entropy calculation is done here
def entropy(df, labels):
    number_of_classes = len(labels)
    p = df / sum(df)
    #ent = -(p) * (np.log(p) / np.log(number_of_classes))
    ent = 1- ((p*p).sum())
    total_ent = ent.sum()
    return total_ent

#Highest gain attribute is calculated and selected
def select_attr(dataset, labels):
    Target_list = pd.DataFrame(dataset['Target'].value_counts(dropna=False))
    ent_tar = entropy(Target_list['Target'], labels)

    if ent_tar == 0:
        return Target_list.index[0], ent_tar

    attributes_list = list(dataset.drop('Target', axis=1).columns)
    attr_gains_list = []
    for attribute in attributes_list:
        grouped_df = dataset.groupby([attribute, 'Target']).size().to_frame(name='size').reset_index()
        attr_ent = 0
        for attr_val in grouped_df[attribute].unique():
            tmp_df = grouped_df[grouped_df[attribute] == attr_val]
            q = tmp_df['size'].sum()
            z = dataset.shape[0]

            s = tmp_df['size'].sum() / dataset.shape[0]

            ent_val = entropy(tmp_df['size'], labels) * s
            attr_ent = attr_ent + ent_val
        gain_of_attr = gain(ent_tar, attr_ent)
        attr_gains_list.append(gain_of_attr)
    max_gain_attr = attributes_list[np.argmax(attr_gains_list)]
    return max_gain_attr, ent_tar


def ID3(dataset, labels, node):
    parent, ent = select_attr(dataset, labels)
    for attr_val in dataset[parent].unique():
        new_dataset = dataset[dataset[parent] == attr_val].drop([parent], axis=1)
        child_node, ent_tar = select_attr(new_dataset, labels)
        if child_node in labels:
            ET.SubElement(node, 'node', {'entropy': str(ent_tar), 'feature': parent, 'value': attr_val}).text = str(child_node)
            continue
        else:
            next_node = ET.SubElement(node, 'node', {'entropy': str(ent_tar), 'feature': parent,  'value': attr_val})
        ID3(new_dataset, labels, next_node)


# def init(input_filename,output_filename):
def init():
    dataset = pd.read_csv('car.csv', delimiter=',')
    dataset
    # def
    # if (dataset['Walc'] + dataset['Dalc']>6):
    #     dataset['alc_prob'] = alc_p
    dataset = dataset.rename(lambda x: 'att' + str(x),axis=1)
    dataset = dataset.rename({dataset.columns[-1]: 'Target'},axis=1)

    labels = dataset['Target'].unique().tolist()
    Target_list = pd.DataFrame(dataset['Target'].value_counts(dropna=False))
    ent_tar = entropy(Target_list['Target'], labels)

    root = ET.Element('tree', {'entropy': str(ent_tar)})

    ID3(dataset, labels, root)
    tree = ET.ElementTree(root)
    tree.write("student_alc.xml")


if __name__ == "__main__":
   # input_filename = None
   # output_filename = None
   # # passing arguments for command line execution
   # opts = getopt.getopt(sys.argv[1:], '', ["data=", "output="])
   # opts = opts[0]
   # # print(opts)
   # for opt, arg in opts:
   #     if opt == '--data':
   #         input_filename = arg
   #     elif opt == '--output':
   #         output_filename = arg
   #
   # if input_filename == None or output_filename == None :
   #     print('Please provide all the inputs: input_filename , output_filename')
   #     exit()
   # init(input_filename,output_filename)
    init()