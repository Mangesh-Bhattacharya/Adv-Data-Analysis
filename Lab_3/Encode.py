import csv
from csv import reader, writer

cols = ["label"]  # list of columns to be encoded
# file containing list of columns to be encoded
with open('/home/mangesh/Downloads/features-list.txt') as f_obj:
    while line := f_obj.readline():  # read line by line from file
        cols.append(line.strip())  # append to list of columns to be encoded

cols.insert(0, 'MD5')  # insert md5sum as first column to be encoded
cols.append('label')  # append label as last column to be encoded

# open dataset.csv in append mode with newline as delimiter
with open('/home/mangesh/Downloads/dataset.csv', 'a', newline='') as f_obj:
    writer_object = writer(f_obj)  # create writer object for dataset.csv
    # write list of columns to be encoded as first row in dataset.csv
    writer_object.writerow(cols)

with open('/home/mangesh/Downloads/features.csv') as f_obj:  # open features.csv in read mode
    reader_object = csv.reader(f_obj)  # create reader object for features.csv
    for row in reader_object:  # read row by row from features.csv
        data = []  # list to store encoded data
        data.append(row[0])  # append md5sum to list of encoded data
        label = row[-1]  # store label in label variable

    if label == 'benignware':  # if label is benignware
        label = '0'  # set label to 0
    elif label == 'malware':  # if label is malware
        label = '1'  # set label to 1
    else:  # if label is neither benignware nor malware
        label == 'unknown'  # set label to unknown
        label = '2'  # set label to 2

    for y in cols[1:-1]:  # iterate over list of columns to be encoded
        if y in row[1:-1]:  # if column is present in row
            binary = '1'  # set binary to 1
        else:  # if column is not present in row
            binary = '0'  # set binary to 0
        data.append(binary)  # append binary to list of encoded data
    data.append(label)  # append label to list of encoded data

    # open dataset.csv in append mode with newline as delimiter
    with open('/home/mangesh/Downloads/dataset_new.csv', 'a', newline='') as f_obj:
        writer_object = writer(f_obj)  # create writer object for dataset.csv
        # write list of encoded data to dataset.csv
        writer_object.writerow(data)
