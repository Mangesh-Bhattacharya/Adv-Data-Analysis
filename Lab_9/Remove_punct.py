# Import Libraries to remove punctuation from CSV file and write to new file
import csv
import string
# Function to replace punctuation (?, !, ., etc) with a space
def remove_punct(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text
# Function to remove punctuation (?, !, ., etc) from a string
def remove_punct2(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Open the file to be read and the file to be written
with open('lab9/Lab9Data.csv', 'r') as read_obj, \
        open('lab9/features.csv', 'w', newline='') as write_obj:
    # Create a csv.reader object from the input file object
    csv_reader = csv.reader(read_obj)
    # Create a csv.writer object from the output file object
    csv_writer = csv.writer(write_obj)
    # Read each row of the input csv file as list
    for row in csv_reader:
        # Pass the list as an argument into the remove_punct() function
        # and join the returned string
        row[0] = remove_punct2(row[0])
        row[1] = remove_punct2(row[1])
        row[2] = remove_punct2(row[2])
        row[3] = remove_punct2(row[3])
        row[4] = remove_punct2(row[4])
        row[5] = remove_punct2(row[5])
        row[6] = remove_punct2(row[6])
        row[7] = remove_punct2(row[7])
        row[8] = remove_punct2(row[8])
        row[9] = remove_punct2(row[9])
        row[10] = remove_punct2(row[10])
        row[11] = remove_punct2(row[11])
        row[12] = remove_punct2(row[12])
        row[13] = remove_punct2(row[13])
        row[14] = remove_punct2(row[14])
        row[15] = remove_punct2(row[15])
        row[16] = remove_punct2(row[16])
        row[17] = remove_punct2(row[17])
        # Append the list to the output file
        csv_writer.writerow(row)

# # Create a header list for the data frame and add it to the data frame
# header = ['Date','Name','Country','BusinessType','BusinessSubType','BreachType','DataType','DataType2','InsideOutside','ThirdParty','ThirdPartyName','TotalAffected','RefPage','UID','StockSymbol','DataRecovered','ConsumerLawsuit','ArrestProsecution']
# print(header)