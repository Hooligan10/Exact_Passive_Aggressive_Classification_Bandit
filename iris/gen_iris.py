# Generating feature vectors and labels from the dataset

data = {}
data['featurelist'] = [] ## Feature vectors
data['truelabel'] = [] ## Ground Truth

f = open('iris.txt','r')
if f.mode == 'r':
	content = f.read()

content = content.strip('\n')
content = content.split('\n')

for line in content:
	line = str(line)
	line = line.split(',')
	arr = []

	for i in range(0,len(line)-1):
		arr.append(float(line[i]))
	data['featurelist'].append(arr)

	if(line[4] == 'Iris-setosa'):
		data['truelabel'].append(0)
	elif(line[4] == 'Iris-versicolor'):
		data['truelabel'].append(1)
	else:
		data['truelabel'].append(2)
