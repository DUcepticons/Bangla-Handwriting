import csv

data=[]
with open('a-1-1.csv') as csvfile:
    data = list(csv.reader(csvfile))[0]
    data = [float(i) for i in data] 
    
