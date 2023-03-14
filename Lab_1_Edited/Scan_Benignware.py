#!/usr/bin/python

import os
import subprocess as sp

files = os.listdir('/home/user/Downloads/malware_data_science/ch8/data/benignware/')

for file in files:
    file = "/home/user/Downloads/malware_data_science/ch8/data/benignware/" + file
    stream = os.popen("md5sum " + file)
    hvalue = stream.read()
    hvalue = hvalue.split()[0]
    proc = sp.Popen(["/home/user/Downloads/capa", "-v", file], stdout=sp.PIPE, universal_newlines=True)
    output, error = proc.communicate()
    layer = output.split("\n\n")
    layer.pop(0)
    feat = []
    feat.append(hvalue)
    feat.append(0)

    for element in layer:
        if element == "":
            continue
        else:
            element = element.split("\n")
            if "(." in element[0]:
                feat.append(element[0])
            elif "(" in element[0]:
                ls_element = element[0].split("(")[0]
            else:
                feat.append(element[0])
    print("Examined files: ", file)
    print(feat)
    with open("/home/user/Downloads/features.csv", "a") as f:
        f.write(str(feat))
        f.write("\n")
    f.close()

# Path: Scan_Malware.py
