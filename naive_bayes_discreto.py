import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

def read_txt_to_dataset(file_name):
    data = pd.read_csv(file_name + "_data.txt", sep=";", header=None)
    target = pd.read_csv(file_name + "_target.txt", sep=";", header=None)
    target_matrix = np.array(target)
    dataset = {'data': np.array(data),
               'target': target_matrix.flatten()
              }
    return dataset

def separate_group(data, target_list, reference):
    '''
    This function is used to separate the "data" into smaller arrays according to the target group choosed in "reference".

    Example:
        data = [[1,2,3,4],[1,2,3,4]]
        target_list = [0,0,1,1]
        reference = 0
        group = separate_group(data, target_list, reference)
    '''
    data_group = np.empty((0,len(data[0])))
    for element in range(len(target_list)):
        if target_list[element] == reference:
            data_group = np.append(data_group, np.asmatrix(data[element]), axis=0)
    return data_group

def classify_data(data, age, income, student, credit, target_matrix):
    '''
    This function returns the most probably result according gaussian naive bayes algorithm.
    It receives the matrices of average and standard deviation.
    Example: 
        data_vector = [1, 2, 3, 4]
        groups_average = [[1,1.5,2,1],[0.5,1,2,4]]
        groups_std_dev = [[1,0.2,0.5,0.6],[1,1.5,2,1]]
        target = classify_data(data_vector, groups_average, groups_std_dev)
    '''

    likelihood = np.ones(age.shape[1])
    for target in range(age.shape[1]):
        likelihood[target] = age[data[0]][target] * income[data[1]][target] * student[data[2]][target] * credit[data[3]][target] * target_matrix[target]
    max_value = np.amax(likelihood)
    index = np.where(likelihood == max_value)
    target = index[0][0]
    return target

def get_look_up_table(dataset, target, unique, column_in_dataset, total_of_targets):
    result = np.zeros((unique,len(total_of_targets)))
    for column in range(len(total_of_targets)):
        group = np.array(separate_group(dataset, target, column))
        for row in range(unique):
            vector = np.count_nonzero(group == row, axis=0)
            result[row][column] = vector[column_in_dataset] / total_of_targets[column]

    return result

def main():

    dataset = read_txt_to_dataset("exercicio")
    age_unique = 3
    income_unique = 3
    student_unique = 2
    credit_unique = 2

    #num_of_classes = len(set(dataset['target']))
    total_of_entry = len(dataset["data"])

    target_yes = np.count_nonzero(dataset['target'] == 1)
    target_no = np.count_nonzero(dataset['target'] == 0)
    total_targets = np.array([target_no, target_yes])

    
    age = get_look_up_table(dataset["data"], dataset['target'], age_unique, 0, total_targets)
    income = get_look_up_table(dataset["data"], dataset['target'], income_unique, 1, total_targets)
    student = get_look_up_table(dataset["data"], dataset['target'], student_unique, 2, total_targets)
    credit = get_look_up_table(dataset["data"], dataset['target'], credit_unique, 3, total_targets)

    targets = np.array([target_no, target_yes]) / total_of_entry

    print()
    print("Age")
    print(age)
    print()
    print("Income")
    print(income)
    print()
    print("Student")
    print(student)
    print()
    print("Credit")
    print(credit)
    print()
    print("Targets")
    print(targets)

    targets_names = ["No", "Yes"]
    again = input("\nDo you wanna try a point? (y/n) ")
    while again == "y":
        point_str = input("Enter a point. Ex. 2;3;8  \nATTENTION: WITHOUT PARENTHEIS!!!\n")
        point_list = point_str.split(";")
        point_int = [int(i) for i in point_list] 
        result = classify_data(point_int, age, income, student, credit, targets)
        print("\nTarget of the inserted point is: " + targets_names[result])
        again = input("\nDo you wanna try another point? (y/n) ")



if __name__ == "__main__":
    main()
