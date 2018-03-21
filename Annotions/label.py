import csv

filename = '/home/qer/PycharmProjects/vgg19/Annotations/label.csv'
w = open('/home/qer/PycharmProjects/vgg19/Annotations/pant_length_labels.csv', 'w', newline='')
writer = csv.writer(w)
datas = []
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1] == 'pant_length_labels':
            datas.append(row)

for row in datas:
    row[2] = [i for i in range(6) if row[2][i] == 'y'][0]

writer.writerows(datas)


