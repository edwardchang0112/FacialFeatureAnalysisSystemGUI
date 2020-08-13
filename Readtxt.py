def readTXTFile(file_name):
    files = []
    print("file_name = ", file_name)
    f = open(file_name+".txt", "r")
    for line in f.readlines():
        line = line.strip('\n')
        files.append(line)
    print("files = ", files)
    return files