# K-means clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipaddress
import csv
import sklearn
from pandas import read_csv
from sklearn.cluster import KMeans

# Read in the data from the CSV file lab8/Log_NB_4.csv with separator and encoding
dataset = read_csv('lab8/Log_NB_4.csv', sep=',', encoding='cp1252')

# Convert IP address with decimal and remove the dot
dataset['srcip'] = dataset['srcip'].astype(str).str.replace('.', '')
print(dataset['srcip'])
dataset['dstip'] = dataset['dstip'].astype(str).str.replace('.', '')
print(dataset['dstip'])

# Replace columns srcip and dstip to converted IP address value in dataset lab8/Log_NB_4.csv
# dataset.replace({'srcip': ip_to_int(srcip), 'dstip': ip_to_int(dstip)}, inplace=True)

# Add converted IP address to list dstip and srcip columns
dstip = []
for dstip in dataset.iloc[:, 2]:
    try:
        dstip = int(ipaddress.ip_address(dstip))
        print(dstip)
        dstip.append(dstip)
    except:
        print(dstip)

# Add converted IP address to list srcip column
srcip = []
for srcip in dataset.iloc[:, 0]:
    try:
        srcip = int(ipaddress.ip_address(srcip))
        srcip.append(srcip)
    except:
        print(srcip)

# Replace the converted IP address to dataset lab8/Log_NB_4.csv
#dataset = dataset.replace({'source_ip': srcip, 'dest_ip': dstip}, inplace=True)

# Print new columns srcip and dstip
# print(dataset['srcip'])
#dataset.loc[:,50] = srcip
# print(dataset['dstip'])
#dataset.loc[:,51] = dstip

# In column proto, replace the value of TCP, UDP, ICMP, ARP, sctp, ospf, rdp, igmp, 3pc, a/n, ipip, pim, rsvp, vrrp, gre, aes-sp3-d, ipcomp, l2tp, pcp, sdrp, skip, st2, sun-nd, vmtp, wesp, xnet, ipip, ipnip, ipnip, pim, rsvp, vrrp, gre, aes-sp3-d, ipcomp, ipv
dataset['proto'] = dataset['proto'].replace(['TCP', 'UDP', 'ICMP', 'ARP', 'sctp', 'ospf', 'rdp', 'igmp', '3pc', 'a/n', 'ipip', 'pim', 'rsvp', 'vrrp', 'gre', 'aes-sp3-d',
                                            'ipcomp', 'l2tp', 'pcp', 'sdrp', 'skip', 'st2', 'sun-nd', 'vmtp', 'wesp', 'xnet', 'ipip', 'ipnip', 'ipnip', 'pim', 'rsvp', 'vrrp', 'gre', 'aes-sp3-d', 'ipcomp'], 1)
print(dataset['proto'])

# In column state, replace the value of FIN, CON, INT, REQ, RST, ACC, PAR, CLO, RSTO, RSTR, NUL, ECN, URH, ospf, ECO, no, URN, ICN, TST, and others to 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, and 20
dataset['state'] = dataset['state'].replace(['FIN', 'CON', 'INT', 'REQ', 'RST', 'ACC', 'PAR', 'CLO', 'RSTO', 'RSTR', 'NUL', 'ECN',
                                            'URH', 'ospf', 'ECO', 'no', 'URN', 'ICN', 'TST', 'others'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
print(dataset['state'])

# In column service, replace the value of http, ftp-data, smtp, dns, ssh, pop3 to 1, 2, 3, 4, 5, and 6
dataset['service'] = dataset['service'].replace(
    ['http', 'ftp-data', 'smtp', 'dns', 'ssh', 'pop3', '-', 'dhcp', 'irc', 'radius', 'snmp', 'ssl'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(dataset['service'])

# In column attack_cat, replace the value of Normal, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms, and other to 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, and 11
dataset['attack_cat'] = dataset['attack_cat'].replace(['Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS',
                                                      'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms', 'other'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
print(dataset['attack_cat'])

# Print to dataset after replace the value of columns proto, state, service, and attack_cat
dataset.to_csv('lab8/Log_NB_4.csv', index=False)

# Drop blank or empty columns
dataset = dataset.dropna(axis=1)

# Dataset Filter by columns sport, dsport, sbytes, and dbytes only
X = dataset.filter(['state', 'service'], axis=1)
# print(X)

# Apply kmeans clustering with 3 different values of k (2,5 and 10) and print the centroids
kmeans = KMeans(n_clusters=3).fit(X)
print('Clusters Assigned: ')
print(kmeans.labels_)
print('Centroids: ')
print(kmeans.cluster_centers_)
plt.scatter(X['state'], X['service'], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], color='black')
plt.title('K-Means Clustering with 3 Clusters')
plt.xlabel('Service')
plt.ylabel('State')
plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3',
           'Centroids'], loc='upper right')
plt.show()

#  Plot the top 5 features bar-graphs for each value of k (2,5 and 10) and different colors for each cluster
plt.bar(X['state'], X['service'], color=['blue', 'orange'])
plt.title('K-Means Clustering with 3 Clusters')
plt.xlabel('Service')
plt.ylabel('State')
plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3',
           'Centroids'], loc='upper right')
plt.show()
