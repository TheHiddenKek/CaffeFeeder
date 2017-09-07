fin = open("train.txt")

fout = open("train.csv", 'w')

Data = [i[:-1].split('\t') for i in fin if i != '\n']

fin.close()

for i in range(len(Data)):
    for j in Data[i]:
        fout.write(j + ',')
    fout.write(str(i//20 + 1) + "\n")

fout.close()

fin = open("test.txt")

fout = open("test.csv", 'w')

Data = [i[:-1].split('\t') for i in fin if i != '\n']

fin.close()

for i in range(len(Data)):
    for j in Data[i]:
        fout.write(j + ',')
    fout.write(str(i//20 + 1) + "\n")

fout.close()
