import csv

'''mylist = [1,1,0,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1,0,0,1,1,1,1,0]
myfile = open("sandeep_Y.csv", 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(mylist)'''

with open('prakhar_Y.csv', 'rb') as csvfile:
	     spamreader = csv.reader(csvfile)
	     for row in spamreader:
			print row
