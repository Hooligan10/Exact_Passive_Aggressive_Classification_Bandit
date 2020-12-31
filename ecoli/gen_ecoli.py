# Generating feature vectors and corresponding labels

f = open('ecoli.txt','r')
data = {}
data['featurelist'] = [] # Feature vectors
data['truelabel'] = [] # Ground Truth

if f.mode == 'r':
	content = f.read()
	content = content.split('\n')
	for j in range(0, len(content)):
		strinter = str(content[j])
		inter = strinter.split(',')
		ar = []
		for i in range(1,len(inter)-1):
			ar.append(float(inter[i]))
		data['featurelist'].append(ar)
		data['truelabel'].append(inter[i+1])
