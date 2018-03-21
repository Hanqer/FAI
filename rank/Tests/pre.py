import csv

filename = '/home/qer/PycharmProjects/vgg19/rank/Tests/question.csv'
## ======== Do not forget to change this. =====##
outputName = 'skirt_length.csv'    # skirt_length
w = open('/home/qer/PycharmProjects/vgg19/rank/Tests/'+outputName, 'w', newline='')
writer = csv.writer(w)
datas = []
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1] == 'skirt_length_labels': # skirt_length_labels
            datas.append([row[0], row[1]])

writer.writerows(datas)

w.close()
