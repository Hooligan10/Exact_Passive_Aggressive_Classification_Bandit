# Generating feature vector and corresponding labels from the dataset

f = open('abalonedata.txt','r')
data = {}
data['featurelist'] = [] # Stores feature vectors
data['truelabel'] = [] # Stores ground truth

for line in f:
	line = line.replace('\n','')
	sp = line.split(',')
	sp1 = sp[1:8]
	
	feature = [float(numeric_string) for numeric_string in sp1]
	true_label = int(sp[8])
	
	data['featurelist'].append(feature[0:8])

	if true_label>=1 and true_label <=7:
		true_label = 0
	elif true_label>=8 and true_label <=14:
		true_label = 1
	elif true_label>=15 and true_label <=21:
		true_label = 2
	else:
		true_label = 3

	data['truelabel'].append(true_label)
