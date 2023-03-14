#!/usr/bin/python

import os  # import the os module
import subprocess as sp  # import the subprocess module as sp

# Path to the directory containing the files to be scanned
# Change this to the path where you have stored the files
files = "/home/user/Downloads/malware_data_science/ch8/data/benignware/agentsvr.exe"
# MD5 hash of the file to be generated using the md5sum command
# run the md5sum command and store the output in stream variable
stream = os.popen("md5sum" + files)
# read the output of the command and store it in Hash Value variable hvalue
hvalue = stream.read()
# Hash value variable contains various numerical patterns and path to the file, we must filter the path and no values
# As per instructions, the benignware is a 0 and malware is a 1
hvalue = hvalue.split()
# print the hash value to the console for debugging
print("Hash Value is: " + str(hvalue))
# run the capa command and store the output in process variable
proc = sp.Popen(["/home/user/Downloads/capa", "-v", files], stdout=sp.PIPE, universal_newlines=True)
# get the output of the command and store it in output variable and error in error variable
output, error = proc.communicate()
# print the output to the console for debugging
print(output)
