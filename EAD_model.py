
# This file implements the training phase of entropy-based anomaly detection model.

# --------------------------- IMPORTS --------------------------------------
import pickle
import random
from operator import itemgetter
from collections import defaultdict, Counter
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# -------------------- FUNCTION IMPLEMENTATIONS -----------------------------

# Function : write_to_excel
# Inputs : numpy matrix, list of component IDs, output filename
# Output :  dataframe
# Description : Auxillary function to convert a numpy matrix to dataframe format
##--------------------------------------------------------------------------------
def write_to_excel(numpy_matrix, id_list, dest_fname):
    df= pd.DataFrame(numpy_matrix, columns= id_list, dtype=float)
    df.to_excel(dest_fname)
    return (df)

           
# Function : plot_heatmap
# Inputs : dataframe
# Output : Heatmap saved
# Description : Plot dataframe as heatmap 
##--------------------------------------------------------------------------------
def plot_heatmap(df):
    plt.figure(figsize = (15,6))
    sns.heatmap(df, cmap="viridis", annot=False, edgecolors='w', linewidths=0.1, linestyle= "dotted");
    #plt.show()
    plt.savefig('heatmap.png')
 

# Function : clean_data
# Inputs : raw source file name, outut file name 
# Output : Pickle file containing processed data saved
# Description : Read raw data from source, process to format
#           (timestamp, ID, payload, is anomaly) and serialise and store as pickle
##--------------------------------------------------------------------------------
def clean_data(src_fname, dest_fname):
    bus_data=[]
    source_file = open(src_fname, "r")
    line_idx=0
    # line format : timestamp, t+id(3)+DLC(1)+payload(max 8)+ignore bits(4)
    for line in source_file:
        line_idx=line_idx+1
        line_data = line.split(",")
        comp_id=line_data[1][1:4]        
        bus_data.append((line_data[0], comp_id, line_data[1][5:-5], False))
    source_file.close()
    #serialise as pickle
    pickle.dump(bus_data, open(dest_fname, "wb"))


# Function : compute_entropy
# Inputs : processed data, lis of ids
# Output : 2D matrix; rows =byte index , col=ID, val=entropy
# Description : for each unique component ID, compute byte wise entropy
# Note: index of rows to follow the id_list order throughout
##--------------------------------------------------------------------------------
def compute_entropy(bus_data, id_list):

    data_group_per_ID = defaultdict(list)

    #matrix to store final entropy of byte per ID
    entropy_matrix = np.full((8 ,len(id_list)), -1.1)

    for (ts,comp_id,data,anomaly) in bus_data:
        data_group_per_ID[comp_id].append([data[i:i + 2] for i in range(0, len(data), 2)])
    
    for key, value in data_group_per_ID.items():
        temp_transpose = list(map(list, zip(*value)))  # [[byte 0 values], [byte 1 values]...]
        
        for byte_val_list in temp_transpose:
            byte_entropy = 0
            length = len(byte_val_list)
            
            counter=Counter(byte_val_list) # determine the freq of each unique byte value
            total= sum(counter.values())
            
            probability = [counter[key]/total for key in counter.keys()]
            byte_entropy = sum([-(x * math.log(x,2)) for x in probability])

            entropy_matrix[temp_transpose.index(byte_val_list)][id_list.index(key)]=byte_entropy

    return (entropy_matrix)


# Function : split_dataset
# Inputs : processed data of format (ts,id,payload,anomaly)
# Output : dataset split into windows of 10000
# Description : Input dataset is split into equal sized windows 
##--------------------------------------------------------------------------------
def split_dataset(bus_data):
    n=10000
    split_bus_data=[bus_data[i:i + n] for i in range(0, len(bus_data), n)]
    return (split_bus_data)


# Function : compute_score
# Inputs : dataset windows, list of IDs, overall entropy matrix 
# Output : 3D matrix; for each window - rows =byte index , col=ID, val=difference in entropy 
# Description : For each window, compute entropy and then the difference with the 
#            overall dataset entropy, denoted as score of the window 
##--------------------------------------------------------------------------------
def compute_score(data_windows, id_list, entropy_matrix):
    len_data_windows = len(data_windows)
    len_id_list = len(id_list)
    
    score_matrix = np.full((len_data_windows ,8 ,len_id_list), np.NaN)

    #for each dataset window, compute entropy
    # and then compare with overall dataset entropy
    for window_idx in range(len_data_windows):
        window_entropy=compute_entropy(data_windows[window_idx], id_list)
        diff_window = entropy_matrix - window_entropy
        score_matrix[window_idx] = diff_window

        df= pd.DataFrame(score_matrix[window_idx], columns= id_list)
        df.to_excel("diff_entropy_matrix_"+str(window_idx)+".xlsx")

    return (score_matrix)


# Function : feature_extract
# Inputs : 3D score matrix of windows
# Output : 2D matrix rows =byte index , col=ID, val=(mean,std) across windows
# Description : Compute the mean and standard deviation per ID per byte of combined window scores
#           This feature will help compute the normal distribution fit of test data
##--------------------------------------------------------------------------------
def feature_extract(score_matrix):
    #compute mean and std with window axis constant
    std_matrix=np.std(score_matrix, axis=0)
    mean_matrix = np.mean(score_matrix, axis=0)
    
    # combine std deviation and mean as a tuple into a numpy array
    arr = np.zeros(mean_matrix.shape, dtype='float,float')
    arr["f0"]=mean_matrix
    arr["f1"]=std_matrix

    return(arr)
    
    
#-------------------------- FUNCTION CALLS-------------------------------------------

#Initial Step - read data from the raw data file and save it in the required format
#Saved as a pkl with format (timestamp, component ID, payload, is anomaly?)
output1= clean_data("train_data.txt", "busData.pkl")

#read the training data stored as a pkl
bus_data = pickle.load( open("busData.pkl", "rb" ) )

# list of unique component Ids in the CAN bus data
id_list=list(set(map(itemgetter(1), bus_data)))

# Compute entropy of dataset
entropy_matrix = compute_entropy(bus_data, id_list)

#write to excel and visualise as heatmap
df = write_to_excel(entropy_matrix, id_list,"entropy_matrix.xlsx")
output3 = plot_heatmap(df)

# Split the training data into training windows
train_windows = split_dataset(bus_data)

#Compute score of each training window
score_matrix = compute_score(train_windows, id_list, entropy_matrix)

#Extract normal distribution feature 
feature_matrix = feature_extract(score_matrix)

#Serialise extracted feature as pickle
pickle.dump(feature_matrix, open("feature_matrix.pkl", "wb"))
