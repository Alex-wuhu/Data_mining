import sys
import pandas as pd
import numpy as np
from math import log2

attribute_names = []
train_dataframe = None
attribute_values_dict = {}


class Node:
    def __init__(self, class_label=None, is_leaf=False):
        self.attribute = []
        self.class_label = class_label
        self.is_leaf = is_leaf
        self.child = {}


def get_attributes_from_train(file_name):
    global attribute_names, train_dataframe, attribute_values_dict
    train_dataframe = pd.read_csv(file_name, sep="\t")
    attribute_names = train_dataframe.columns.values.tolist()
    attribute_values_dict = {attribute: train_dataframe[attribute].unique() for attribute in train_dataframe.columns}


def info(data_samples):
    entropy = 0
    data_samples_len = len(data_samples)
    classified_samples = classify_samples(data_samples)
    for class_label in classified_samples:
        probability = float(classified_samples[class_label]) / data_samples_len
        entropy -= probability * log2(probability + 1e-10)
    return entropy


def gain_ratio(data_samples, attribute):
    global attribute_values_dict
    entropy = 0
    split_info = 0
    base_entropy = info(data_samples)
    for value in attribute_values_dict[attribute]:
        sorted_data_samples = data_samples[data_samples[attribute] == value]
        info_a = info(sorted_data_samples)
        entropy += info_a * (len(sorted_data_samples) / len(data_samples))
        split_info -=(len(sorted_data_samples)/len(data_samples)) *log2((len(sorted_data_samples) + 1e-10) / (len(data_samples) + 1e-10))
    if split_info ==0:
        return 0;
    return (base_entropy - entropy)/split_info


#divide data sets by labels
def classify_samples(data_samples):
    classified_sample_dict = {}
    for sample in data_samples.values:
        class_label = sample[-1]
        if class_label not in classified_sample_dict:
            classified_sample_dict[class_label] = 1
        else:
            classified_sample_dict[class_label] += 1
    return classified_sample_dict

#identify by tree node
def search_decision_tree(tree, data):
    node = tree

    data_attr_value = data[node.attribute[-1]]
    while not node.is_leaf:
        node = node.child[data_attr_value]
        if len(node.attribute) != 0:
            data_attr_value = data[node.attribute[-1]]
    if type(node.class_label) == np.ndarray:
        return node.class_label[0]
    return node.class_label


def classify(tree, data_samples):
    class_label = [search_decision_tree(tree, data_samples.loc[sample_idx]) for sample_idx in range(len(data_samples))]
    data_samples[attribute_names[-1]] = class_label
    return data_samples


def make_node(data_samples, used_attribute):
    global attribute_names, attribute_values_dict, train_dataframe
    if data_samples[attribute_names[-1]].nunique() == 1:
        class_label = data_samples[attribute_names[-1]].unique()
        return Node(class_label, True)

    if used_attribute is not None and len(used_attribute) == len(attribute_names)-1:
        class_dict = classify_samples(data_samples)
        class_label = max(class_dict, key=class_dict.get)
        return Node(class_label, True)


    info_gain_attribute_dict = {attribute: gain_ratio(data_samples, attribute)
                                for attribute in data_samples.columns[:-1]}
    selected_attribute = max(info_gain_attribute_dict, key=info_gain_attribute_dict.get)
    base_class_dict = classify_samples(data_samples)
    base_class_label = max(base_class_dict, key=base_class_dict.get)
    node = Node(base_class_label)
    if used_attribute is not None:
        node.attribute.extend(used_attribute)
    node.attribute.append(selected_attribute)

    attribute_values = attribute_values_dict[selected_attribute]
    for attr_value in attribute_values:
        filtered_data = data_samples[data_samples[selected_attribute].isin([attr_value])]
        if len(filtered_data) >= 1:
            node.child[attr_value] = make_node(filtered_data, node.attribute)
        else:
            class_dict = classify_samples(data_samples)
            class_label = max(class_dict, key=class_dict.get)
            node.child[attr_value] = Node(class_label, True)
    return node


def main():
    global train_dataframe
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    get_attributes_from_train(train_file_name)

    decision_tree = make_node(train_dataframe, None)

    test_data_frame = pd.read_csv(test_file_name, sep='\t')
    classify_test_data_frame = classify(decision_tree, test_data_frame)
    classify_test_data_frame.to_csv(output_file_name, index=False, sep='\t')


if __name__ == "__main__":
    main()  