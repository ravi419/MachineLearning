# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 19:07:20 2018

@author: HP
"""
import numpy, pandas
import xml.etree.cElementTree as CET
import sys, getopt


def gain_calc(Intial_entropy, Attribute_entropy):
    gain = Intial_entropy - Attribute_entropy
    return gain


def entropy_calc(df, Class_list):
    number_of_classes = len(Class_list)
    prob = df / sum(df)
    ent = -prob * (numpy.log(prob) / numpy.log(number_of_classes))
    total_ent = ent.sum()
    return total_ent


def Node_selection(reader, Class_list):
    Target_list = pandas.DataFrame(reader['Class'].value_counts())
    Intial_entropy = entropy_calc(Target_list['Class'], Class_list)

    if Intial_entropy == 0:
        return Target_list.index[0], Intial_entropy

    attributes_list = list(reader.drop('Class', axis=1).columns)
    attr_gains_list = []
    for attribute in attributes_list:
        grouped_df = reader.groupby([attribute, 'Class']).size().to_frame(name='size').reset_index()
        Attribute_entropy = 0
        for attr_val in grouped_df[attribute].unique():
            tmp_df = grouped_df[grouped_df[attribute] == attr_val]
            s = tmp_df['size'].sum() / reader.shape[0]
            Entropy_value = entropy_calc(tmp_df['size'], Class_list) * s
            Attribute_entropy = Attribute_entropy + Entropy_value
        gain_of_attr = gain_calc(Intial_entropy, Attribute_entropy)
        attr_gains_list.append(gain_of_attr)
    max_gain_attr = attributes_list[numpy.argmax(attr_gains_list)]
    return max_gain_attr, Intial_entropy


def ID3(reader, Class_list, root):
    parent, entropy = Node_selection(reader, Class_list)
    for attr_val in reader[parent].unique():
        df = reader[reader[parent] == attr_val].drop([parent], axis=1)
        Child_node, ent_tar = Node_selection(df, Class_list)
        if Child_node in Class_list:
            CET.SubElement(root, 'node', {'entropy': str(ent_tar), 'feature': parent, 'value': attr_val}).text = str(
                Child_node)
            continue
        else:
            next_node = CET.SubElement(root, 'node', {'entropy': str(ent_tar), 'feature': parent, 'value': attr_val})
        ID3(df, Class_list, next_node)


# def init():
def init(input_filename, output_filename):
    reader = pandas.read_csv(input_filename, delimiter=',', header=None)
    # reader = pandas.read_csv('car.csv', delimiter=',', header=None)
    reader = reader.rename(lambda x: 'attr' + str(x), axis=1)
    reader = reader.rename({reader.columns[-1]: 'Class'}, axis=1)

    Class_list = reader['Class'].unique().tolist()
    Target_list = pandas.DataFrame(reader['Class'].value_counts())
    Intial_Entropy = entropy_calc(Target_list['Class'], Class_list)

    root = CET.Element('tree', {'entropy': str(Intial_Entropy)})

    ID3(reader, Class_list, root)
    tree = CET.ElementTree(root)
    tree.write(output_filename)
    # tree.write('car_sol.xml')


if __name__ == "__main__":
    input_filename = None
    output_filename = None
    opts = getopt.getopt(sys.argv[1:], '', ["data=", "output="])
    opts = opts[0]
    for opt, arg in opts:
        if opt == '--data':
            input_filename = arg
        elif opt == '--output':
            output_filename = arg

    if input_filename is None or output_filename is None:
        print('Please provide all the inputs: input_filename , output_filename')
        exit()
    init(input_filename, output_filename)
    # init()
