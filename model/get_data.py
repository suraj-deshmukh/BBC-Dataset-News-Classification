import os
import pandas as pd
from sklearn.model_selection import train_test_split

data_folder = "../dataset/data_files"
folders = ["business","entertainment","politics","sport","tech"]

os.chdir(data_folder)

x = []
y = []

for i in folders:
    files = os.listdir(i)
    for text_file in files:
        file_path = i + "/" +text_file
        print "reading file:", file_path
        with open(file_path) as f:
            data = f.readlines()
        data = ' '.join(data)
        x.append(data)
        y.append(i)
   
data = {'news': x, 'type': y}       
df = pd.DataFrame(data)
print 'writing csv flie ...'
df.to_csv('../dataset.csv', index=False)
