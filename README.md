# BBC-Dataset-News-Classification

Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.

Class Labels: 5 (business, entertainment, politics, sport, tech)

# Dataset Discription: 

[BBC Datasets Descrition](http://mlg.ucd.ie/datasets/bbc.html) 

[Dataset](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip)

# Files Description
* dataset/data_files: Data folders each containing several news txt files

* dataset/dataset.csv: csv file containing "news" and "type" as columns. "news" column represent news article and "type" represents news category among business, entertainment, politics, sport, tech.

* model/get_data.py: To gather all txt files into one csv file contianing two columns("news","type"). After successfull execution it will create dataset.csv file in dataset folder. 

* model/model.py: preprocessing, tf-idf feature extraction and model buildind and evaluation stuff

* model/test.ipynb: jupyter notebook 


# Method

Divided the feature extracted dataset into two parts train and test set. Train set contains 1780 examples and Test set contains 445 examples. 

# Result

Below table shows the result on test set

Accuracy | Value
--------- | ---------
Kappa | 0.9461
Accuracy | 0.9573
