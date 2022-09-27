import csv

lists = ["HELLO","HI"]

with open('test.csv',"w+",newline="") as f:
    writer = csv.writer(f, delimiter="\n")
    writer.writerows(lists)
    
    

with open('label.csv', encoding= 'utf-8-sig') as f:
    label = csv.reader(f)
    label = [row[0] for row in label]
    print(label)