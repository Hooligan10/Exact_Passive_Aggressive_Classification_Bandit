# Generating Feature vectors and labels for the dataset

f = open('sat.txt','r')
data = {}
data['featurelist']=[] #Feature vectors
data['truelabel']=[] # Labels
if f.mode == 'r':
	content = f.read()
	content = content.strip('\n')
	inter = content.split('\n')
	for i in range(0,len(inter)):
		strinter = str(inter[i])
		temp = strinter.split(' ')
		ar = []
		for j in range(0,len(temp)-1):
			ar.append(int(temp[j]))
		data['featurelist'].append(ar)
		data['truelabel'].append(int(temp[j+1]))
