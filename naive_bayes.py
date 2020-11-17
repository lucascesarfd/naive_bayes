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

def normal_distribution(data, average, deviation):
    '''
    This function evaluates the normal distribution and returns the result.
    Example:
        data = 1.5
        average = 1
        deviation = 0.28
        result = normal_distribution(data, average, deviation)
    '''
    value = np.exp(-1*np.power((data-average),2)/(2*np.power(deviation,2))) / (np.power(2*np.pi,0.5)*deviation)
    #print("VALUE:")
    #print(value)
    return value

def classify_data(data_vector, groups_average, groups_std_dev):
    '''
    This function returns the most probably result according gaussian naive bayes algorithm.
    It receives the matrices of average and standard deviation.
    Example: 
        data_vector = [1, 2, 3, 4]
        groups_average = [[1,1.5,2,1],[0.5,1,2,4]]
        groups_std_dev = [[1,0.2,0.5,0.6],[1,1.5,2,1]]
        target = classify_data(data_vector, groups_average, groups_std_dev)
    '''
    likelihood = np.ones(groups_average.shape[0])
    for target in range(groups_average.shape[0]):
        for entry in range(groups_average.shape[1]):
            likelihood[target] = likelihood[target] * normal_distribution(data_vector[entry], groups_average[target,entry], groups_std_dev[target,entry])
    max_value = np.amax(likelihood)
    index = np.where(likelihood == max_value)
    target = index[0][0]
    return target

def main():
    colors = ['red', 'blue', 'green', 'orange', 'violet', "olive", "cyan", "gray", "darkviolet"]

    #dataset = read_txt_to_dataset("haberman")
    dataset = datasets.load_iris()
    #dataset = {'data': np.array([[4, 2], [2, 4], [2, 3], [3, 6], [4, 4], [9, 10], [6, 8], [9, 5], [8, 7], [10, 8]]),
    #           'target': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    #          }
    num_of_classes = len(set(dataset['target']))
    groups_std = np.empty((0,len(dataset["data"][0])))
    groups_avg = np.empty((0,len(dataset["data"][0])))

    for group_num in range(num_of_classes):
        group = separate_group(dataset["data"], dataset["target"], group_num)
        groups_avg = np.append(groups_avg, np.average(group, axis=0), axis=0)
        groups_std = np.append(groups_std, np.std(group, axis=0, dtype=np.float64), axis=0)

    result = np.empty(0)
    for element in range(len(dataset["data"])):
        result = np.append(result, classify_data(dataset["data"][element], groups_avg, groups_std))

    print("Groups average Matrix: ")
    print(groups_avg)
    print()
    print("Groups standard deviation Matrix: ")
    print(groups_std)
    print()
    print("Result: ")
    print(result)

    # Separate the groups/classes
    for group_num in range(len(dataset["data"][0])):
        group = separate_group(dataset["data"], result, group_num)
        group_transpose = np.transpose(group)

        # Add group to final plot
        plt.scatter(np.array(group_transpose[0]), np.array(group_transpose[1]), c=colors[group_num])

    plt.show()



if __name__ == "__main__":
    main()