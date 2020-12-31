import matplotlib.pyplot as plt
import os
import random
import datetime
import numpy as np
from numpy import matrix
import math


SYNSEP_CATEGORY_MAPPING = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class Banditron:

	def __init__(self):
		self.gamma = 0.015
		self.dict_length = 100
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0

	def init_weights(self):
		weights = []
		for i in range(0,len(SYNSEP_CATEGORY_MAPPING)):
			weights.append([0.0] * self.dict_length)
		return weights

	def predict_label(self, feature_vectors):
		max = 0.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]
			if total >= max:
				max = total
				label = i
		return label

	def calc_probabilities(self, calculated_label):
		probabilities = [0] * len(self.weights)
		for i in range(0,len(probabilities)):
			probabilities[i] = self.gamma/len(self.weights)
			if i == calculated_label:
				probabilities[i] += (1 - self.gamma)
		return probabilities

	def random_sample(self, probabilities):
		number = random.random() * sum(probabilities)
		for i in range(0,len(probabilities)):
			if number < probabilities[i]:
				return i
				break
			number -= probabilities[i]
		return len(probabilities)-1


	def update_weights(self, update_matrix):
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]

	def get_update_matrix(self, feature_vectors, calculated_label, predicted_label, true_label, probabilities):
		update_matrix = self.init_weights()
		for i in range(0,len(update_matrix)):
			left = 0.0
			right = 0.0
			if true_label == predicted_label and predicted_label == i:
				left = 1/probabilities[i]
			if calculated_label == i:
				right = 1.0
			for j in range(0,len(feature_vectors)):
				update_matrix[i][j] = feature_vectors[j] * (left - right)
		return update_matrix

	def run(self, feature_vectors, true_label):
		self.number_of_rounds += 1.0
		calculated_label = self.predict_label(feature_vectors)
		probabilities = self.calc_probabilities(calculated_label)
		predicted_label = self.random_sample(probabilities)
		if true_label == predicted_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds

		update_matrix = self.get_update_matrix(feature_vectors, calculated_label, predicted_label, true_label, probabilities)
		self.update_weights(update_matrix)


class EPABF:
	def __init__(self):
		self.gamma = 0.0001
		self.dict_length = 100
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0

	def init_weights(self):
		weights = []
		for i in range(0,len(SYNSEP_CATEGORY_MAPPING)):
			weights.append([0.0] * self.dict_length)
		return weights

	def predict_label(self, feature_vectors):
		max = -99999999.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]
			if total >= max:
				max = total
				label = i
		return label

	def calc_probabilities(self, calculated_label):
		probabilities = [0.0] * len(self.weights)
		for i in range(0,len(probabilities)):
			probabilities[i] = self.gamma/len(self.weights)
			if i == calculated_label:
				probabilities[i] += (1 - self.gamma)
		return probabilities

	def random_sample(self, probabilities):
		number = random.random() * sum(probabilities)
		for i in range(0,len(probabilities)):
			if number < probabilities[i]:
				return i
				break
			number -= probabilities[i]
		return len(probabilities)-1


	def update_weights(self, update_matrix):
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]

	def get_update_matrix(self, feature_vectors, calculated_label, predicted_label, true_label, probabilities,support_set,l_tilde,labeltil):
		update_matrix = self.init_weights()
		step_lambda = [0.0 for x in range(0,len(update_matrix))] ## Lambdas for each label
		
		total_support_class_loss = 0.0
		
		modxsq = 0.0
		sum_step_lambda = 0.0
		

		if true_label == predicted_label:
			a = 1/probabilities[predicted_label]
		else:
			a = 0.0

		for i in range(0,len(support_set)):
			total_support_class_loss += l_tilde[support_set[i]]
		

		### Calculating ||X||^2
		for j in range(0,len(feature_vectors)):
			modxsq += feature_vectors[j]*feature_vectors[j]

		### Determining lambdas for each support class label, non support class has lambda = 0
		if predicted_label in support_set:
			for r in range(0,len(support_set)):
				step_lambda[support_set[r]] = (l_tilde[support_set[r]] + ((a*l_tilde[predicted_label])/(1+(len(support_set)*a*a)-a)) - ((a*a*total_support_class_loss)/(1+(len(support_set)*a*a)-a)))/modxsq
		
		else:
			for r in range(0,len(support_set)):
				step_lambda[support_set[r]] = (l_tilde[support_set[r]]  - ((a*a*total_support_class_loss)/(1+(len(support_set)*a*a))))/modxsq

		### Calculating the sum of all lambdas
		for j in range(0,len(step_lambda)):
			sum_step_lambda += step_lambda[j]

		### Determining the weight update corresponding to each label
		for i in range(0,len(update_matrix)):
			### if i is not predicted label(y_tilde), weight update:
			### w(i) = w_t(i) - lambda_i*(x_t)
			if i != predicted_label:
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = -feature_vectors[j] * step_lambda[i]
					
			### if i is predicted label(y_tilde), weight update:
			### w(y_tilde) = w_t(y_tilde) - sum(lambdas)*(x_t)*(1[y_tilde = y_t]/prob(y_tilde)-1)
			else:                
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = feature_vectors[j] * (sum_step_lambda * (a) - step_lambda[predicted_label])


		return update_matrix

	def run(self, feature_vectors, true_label):
		self.number_of_rounds += 1.0
		calculated_label = self.predict_label(feature_vectors)

		probabilities = self.calc_probabilities(calculated_label)

		predicted_label = self.random_sample(probabilities)

		if true_label == predicted_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		

		### Calculating l_tilde for each label
		l_tilde = [-1.0 for x in range(0,len(self.weights))]
		lt_tilde = [-1.0 for x in range(0,len(self.weights))]
		labeltil = [-1 for x in range(0,len(self.weights))]
		loss_pr = 0.0

		#Calculating (w_(y_tilde))*x*1[y_t = yt]/P(y_tilde)
		if true_label == predicted_label:
			for eachVector in range(0,len(feature_vectors)):
				loss_pr += feature_vectors[eachVector]*self.weights[predicted_label][eachVector]
			loss_pr = loss_pr*(1/probabilities[predicted_label])

		# loss_r is (w_r)*x
		for r in range(0,len(self.weights)):
			loss_r = 0.0
			for eachVector in range(0,len(feature_vectors)):
				loss_r += feature_vectors[eachVector]*self.weights[r][eachVector]

			l_tilde[r] = max(0.0, 1 - loss_pr + loss_r)
			lt_tilde[r] = max(0.0, 1 - loss_pr + loss_r)
			labeltil[r] = r
		
		modx = 0.0
		for j in range(0,len(feature_vectors)):
			modx += feature_vectors[j]*feature_vectors[j]
		### Sorting in Decreasing order the l_tildex and corresponding label
		for i in range(len(self.weights)):
			for j in range(0, len(self.weights)-i-1):
				if lt_tilde[j] < lt_tilde[j+1]:
					lt_tilde[j], lt_tilde[j+1] = lt_tilde[j+1], lt_tilde[j]
					labeltil[j], labeltil[j+1] = labeltil[j+1], labeltil[j]

		### If predicted label equals the true label, then a = 1/probability[predicted_label]
		### else 0
		if true_label == predicted_label:
			a = 1/probabilities[predicted_label]
		else:
			a = 0.0

		support_set = []
		
		cm_loss = 0.0 # cumulative loss
		
		### Determining Support Classes for EPABF
		mods = len(self.weights)
		if(lt_tilde[0] != 0.0):
			support_set.append(labeltil[0])
			j = 1
			cm_loss += l_tilde[labeltil[j-1]]
			while(j!= mods):
				if predicted_label in support_set:
					if a == 0.0:
						if(l_tilde[labeltil[j]] != 0.0):
							if(l_tilde[predicted_label] != 0.0):
								support_set.append(labeltil[j])
						j += 1                    
					else:
						rhs = (((1+(j*a*a)-a)/(a*a))*l_tilde[labeltil[j]]) + ((l_tilde[predicted_label])/a)
						if(cm_loss < rhs):
							support_set.append(labeltil[j])
							cm_loss += l_tilde[labeltil[j]]
							j += 1
						else:
							break
				else:
					if a == 0.0:
						if(l_tilde[labeltil[j]] != 0.0):
							support_set.append(labeltil[j])
						j += 1                    
					else:
						rhs = (((1+(j*a*a))/(a*a))*l_tilde[labeltil[j]])
						if(cm_loss < rhs):
							support_set.append(labeltil[j])
							cm_loss += l_tilde[labeltil[j]]
							j += 1
						else:
							break

		update_matrix = self.get_update_matrix(feature_vectors, calculated_label, predicted_label, true_label, probabilities,support_set,l_tilde,labeltil)
		self.update_weights(update_matrix)


		
class EPABF1:

	def __init__(self):
		self.gamma = 0.0001
		self.dict_length = 100
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0

	def init_weights(self):
		weights = []
		for i in range(0,len(SYNSEP_CATEGORY_MAPPING)):
			weights.append([0.0] * self.dict_length)
		return weights

	def predict_label(self, feature_vectors):
		max = -99999999.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]
			# print("For label %s , we have w_i*x as %s" %(str(i),str(total)))
			if total >= max:
				max = total
				label = i
		return label

	def calc_probabilities(self, calculated_label):
		probabilities = [0.0] * len(self.weights)
		for i in range(0,len(probabilities)):
			probabilities[i] = self.gamma/len(self.weights)
			if i == calculated_label:
				probabilities[i] += (1 - self.gamma)
		return probabilities

	def random_sample(self, probabilities):
		number = random.random() * sum(probabilities)
		for i in range(0,len(probabilities)):
			if number < probabilities[i]:
				return i
				break
			number -= probabilities[i]
		return len(probabilities)-1

	def update_weights(self, update_matrix):
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]

	def get_update_matrix(self, feature_vectors, calculated_label, predicted_label, true_label, probabilities,support_set1,l_tilde,labeltil,step_lambda1,c=0.1):
		update_matrix = self.init_weights()
	
		total_support_class_loss1 = 0.0
	
		modxsq = 0.0
		sum_step_lambda1 = 0.0
		

		if true_label == predicted_label:
			a = 1/probabilities[predicted_label]
		else:
			a = 0.0

		for i in range(0,len(support_set1)):
			total_support_class_loss1 += l_tilde[support_set1[i]]

		

		### Calculating ||X||^2
		for j in range(0,len(feature_vectors)):
			modxsq += feature_vectors[j]*feature_vectors[j]

		if predicted_label in support_set1:
			for r in range(0,len(support_set1)):
				temp1 = (l_tilde[support_set1[r]] + ((a*l_tilde[predicted_label])/(1+(len(support_set1)*a*a)-a)) - ((a*a*total_support_class_loss1)/(1+(len(support_set1)*a*a)-a)))/modxsq
				step_lambda1[support_set1[r]] = min(c,temp1)

		else:
			for r in range(0,len(support_set1)):
				temp1 = (l_tilde[support_set1[r]] - ((a*a*total_support_class_loss1)/(1+(len(support_set1)*a*a))))/modxsq
				step_lambda1[support_set1[r]] = min(c,temp1)			

		# ### Calculating the sum of all lambdas
		for j in range(0,len(step_lambda1)):
			sum_step_lambda1 += step_lambda1[j]

		### Determining the weight update corresponding to each label
		for i in range(0,len(update_matrix)):
			### if i is not predicted label(y_tilde), weight update:
			### w(i) = w_t(i) - lambda_i*(x_t)
			if i != predicted_label:
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = -feature_vectors[j] * step_lambda1[i]
					
			### if i is predicted label(y_tilde), weight update:
			### w(y_tilde) = w_t(y_tilde) - sum(lambdas)*(x_t)*(1[y_tilde = y_t]/prob(y_tilde)-1)
			else:                
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = feature_vectors[j] * (sum_step_lambda1 * (a) - step_lambda1[predicted_label])


		return update_matrix
		

	def run(self, feature_vectors, true_label,c=0.1):
		self.number_of_rounds += 1.0
		calculated_label1 = self.predict_label(feature_vectors)

		probabilities = self.calc_probabilities(calculated_label1)
		predicted_label = self.random_sample(probabilities)
		
		if true_label == predicted_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		

		### Calculating l_tilde for each label
		l_tilde = [-1.0 for x in range(0,len(self.weights))]
		lt_tilde = [-1.0 for x in range(0,len(self.weights))]
		labeltil = [-1 for x in range(0,len(self.weights))]
		loss_pr = 0.0

		#Calculating (w_(y_tilde))*x*1[y_t = yt]/P(y_tilde)
		if true_label == predicted_label:
			for eachVector in range(0,len(feature_vectors)):
				loss_pr += feature_vectors[eachVector]*self.weights[predicted_label][eachVector]
			loss_pr = loss_pr*(1/probabilities[predicted_label])

		# loss_r is (w_r)*x
		for r in range(0,len(self.weights)):
			loss_r = 0.0
			for eachVector in range(0,len(feature_vectors)):
				loss_r += feature_vectors[eachVector]*self.weights[r][eachVector]
				# print("r is: %d" %int(r))
				# print(loss_r)
			l_tilde[r] = max(0.0, 1 - loss_pr + loss_r)
			lt_tilde[r] = max(0.0, 1 - loss_pr + loss_r)
			labeltil[r] = r
		
		
		modx = 0.0
		for j in range(0,len(feature_vectors)):
			modx += feature_vectors[j]*feature_vectors[j]
		
		### Sorting in Decreasing order the l_tildex and corresponding label
		for i in range(len(self.weights)):
			for j in range(0, len(self.weights)-i-1):
				if lt_tilde[j] < lt_tilde[j+1]:
					lt_tilde[j], lt_tilde[j+1] = lt_tilde[j+1], lt_tilde[j]
					labeltil[j], labeltil[j+1] = labeltil[j+1], labeltil[j]

		### If predicted label equals the true label, then a = 1/probability[predicted_label]
		### else 0
		if true_label == predicted_label:
			a = 1/probabilities[predicted_label]
		else:
			a = 0.0

		# Determining Support class set for EPABF-I
		support_set1 = []
		step_lambda1 = [0.0 for x in range(0,len(self.weights))]
		step_lambda1_copy = [0.0 for x in range(0,len(self.weights))]
		total_support_class_loss1 = 0.0
		cm_loss1 = 0.0
		flag = 0
		count = 0
		mods = len(self.weights)
		
		if(lt_tilde[0] != 0.0):
			support_set1.append(labeltil[0])
			j = 1
			total_support_class_loss1 += l_tilde[labeltil[0]]
			cm_loss1 += l_tilde[labeltil[0]]

			for i in range(0,len(self.weights)):
				temp1 = (l_tilde[labeltil[i]] + ((a*l_tilde[predicted_label])/(1+(len(support_set1)*a*a)-a)) - ((a*a*total_support_class_loss1)/(1+(len(support_set1)*a*a)-a)))/modx
				
				if min(c,temp1) > 0:
					step_lambda1[labeltil[i]] = min(c,temp1)
				else:
					step_lambda1[labeltil[i]] = 0.0
			
			while(j!= mods):
				
				if predicted_label in support_set1:
					temp1 = (l_tilde[labeltil[j]] + ((a*l_tilde[predicted_label])/(1+(len(support_set1)*a*a)-a)) - ((a*a*total_support_class_loss1)/(1+(len(support_set1)*a*a)-a)))/modx

				else:
					temp1 = (l_tilde[labeltil[j]] - ((a*a*total_support_class_loss1)/(1+(len(support_set1)*a*a))))/modx

				if min(c,temp1) > 0:
					step_lambda1[labeltil[j]] = min(c,temp1)
				else:
					step_lambda1[labeltil[j]] = 0.0


				if step_lambda1[labeltil[j]] > 0:
					if labeltil[j] not in support_set1:
						support_set1.append(labeltil[j])
						total_support_class_loss1 += l_tilde[labeltil[j]]
				else:
					if labeltil[j] in support_set1:
						support_set1.remove(labeltil[j])
						total_support_class_loss1 -= l_tilde[labeltil[j]]
						step_lambda1[labeltil[j]] = 0.0
				
				if j == mods-1:
					for i in range(0,mods):
						if step_lambda1[i] == step_lambda1_copy[i]:
							count += 1
					if count == mods:
						break
					else:
						count = 0
						j = 0						
						for i in range(0,mods):
							step_lambda1_copy[i] = step_lambda1[i]

				else:
					j += 1

		update_matrix = self.get_update_matrix(feature_vectors, calculated_label1, predicted_label, true_label, probabilities,support_set1,l_tilde,labeltil,step_lambda1)
		self.update_weights(update_matrix)


class EPABF2:

	def __init__(self):
		self.gamma = 0.0001
		self.dict_length = 100
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0

	def init_weights(self):
		weights = []
		for i in range(0,len(SYNSEP_CATEGORY_MAPPING)):
			weights.append([0.0] * self.dict_length)
		return weights

	def predict_label(self, feature_vectors):
		max = -99999999.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]
			if total >= max:
				max = total
				label = i
		return label

	def calc_probabilities(self, calculated_label):
		probabilities = [0.0] * len(self.weights)
		for i in range(0,len(probabilities)):
			probabilities[i] = self.gamma/len(self.weights)
			if i == calculated_label:
				probabilities[i] += (1 - self.gamma)
		return probabilities

	def random_sample(self, probabilities):
		number = random.random() * sum(probabilities)
		for i in range(0,len(probabilities)):
			if number < probabilities[i]:
				return i
				break
			number -= probabilities[i]
		return len(probabilities)-1

	def update_weights(self, update_matrix):
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]


	def get_update_matrix(self, feature_vectors, calculated_label, predicted_label, true_label, probabilities,support_set2,l_tilde,labeltil,c=0.1):
		update_matrix = self.init_weights()
		step_lambda2 = [0.0 for x in range(0,len(update_matrix))] ## Lambdas for each label
		total_support_class_loss2 = 0.0
		modxsq = 0.0
		sum_step_lambda2 = 0.0

		if true_label == predicted_label:
			a = 1/probabilities[predicted_label]
		else:
			a = 0.0

		for i in range(0,len(support_set2)):
			total_support_class_loss2 += l_tilde[support_set2[i]]

		### Calculating ||X||^2
		for j in range(0,len(feature_vectors)):
			modxsq += feature_vectors[j]*feature_vectors[j]

		### Determining lambdas for each support class label, non support class has lambda = 0
		if predicted_label in support_set2:
			num = (a*l_tilde[predicted_label]) - (((a*a)*total_support_class_loss2))
			den = ((a*a*len(support_set2)) + 1 - a + (1/(modxsq*2*c)))
			for r in range(0,len(support_set2)):
				step_lambda2[support_set2[r]] = ((l_tilde[support_set2[r]] + (num/den))/(modxsq*(1+(1/(modxsq*2*c)))))
		else:
			num = - (((a*a)*total_support_class_loss2))
			den = ((a*a*len(support_set2)) + 1 + (1/(modxsq*2*c)))
			for r in range(0,len(support_set2)):
				step_lambda2[support_set2[r]] = ((l_tilde[support_set2[r]] + (num/den))/(modxsq*(1+(1/(modxsq*2*c)))))

		### Calculating the sum of all lambdas
		for j in range(0,len(step_lambda2)):
			sum_step_lambda2 += step_lambda2[j]
		
		### Determining the weight update corresponding to each label
		for i in range(0,len(update_matrix)):
			### if i is not predicted label(y_tilde), weight update:
			### w(i) = w_t(i) - lambda_i*(x_t)
			if i != predicted_label:
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = -feature_vectors[j] * step_lambda2[i]
					
			### if i is predicted label(y_tilde), weight update:
			### w(y_tilde) = w_t(y_tilde) - sum(lambdas)*(x_t)*(1[y_tilde = y_t]/prob(y_tilde)-1)
			else:                
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = feature_vectors[j] * ((sum_step_lambda2*a) - step_lambda2[predicted_label])


		return update_matrix


	def run(self, feature_vectors, true_label,c=0.1):
		self.number_of_rounds += 1.0
		calculated_label2 = self.predict_label(feature_vectors)
		probabilities = self.calc_probabilities(calculated_label2)
		predicted_label = self.random_sample(probabilities)
		
		if true_label == predicted_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		

		### Calculating l_tilde for each label
		l_tilde = [-1.0 for x in range(0,len(self.weights))]
		lt_tilde = [-1.0 for x in range(0,len(self.weights))]
		labeltil = [-1 for x in range(0,len(self.weights))]
		loss_pr = 0.0

		#Calculating (w_(y_tilde))*x*1[y_t = yt]/P(y_tilde)
		if true_label == predicted_label:
			for eachVector in range(0,len(feature_vectors)):
				loss_pr += feature_vectors[eachVector]*self.weights[predicted_label][eachVector]
			loss_pr = loss_pr*(1/probabilities[predicted_label])

		# loss_r is (w_r)*x
		for r in range(0,len(self.weights)):
			loss_r = 0.0
			for eachVector in range(0,len(feature_vectors)):
				loss_r += feature_vectors[eachVector]*self.weights[r][eachVector]
			l_tilde[r] = max(0.0, 1 - loss_pr + loss_r)
			lt_tilde[r] = max(0.0, 1 - loss_pr + loss_r)
			labeltil[r] = r
		
		
		modx = 0.0
		for j in range(0,len(feature_vectors)):
			modx += feature_vectors[j]*feature_vectors[j]

		### Sorting in Decreasing order the l_tildex and corresponding label
		for i in range(len(self.weights)):
			for j in range(0, len(self.weights)-i-1):
				if lt_tilde[j] < lt_tilde[j+1]:
					lt_tilde[j], lt_tilde[j+1] = lt_tilde[j+1], lt_tilde[j]
					labeltil[j], labeltil[j+1] = labeltil[j+1], labeltil[j]

		### If predicted label equals the true label, then a = 1/probability[predicted_label]
		### else 0
		if true_label == predicted_label:
			a = 1/probabilities[predicted_label]
		else:
			a = 0.0

		support_set2 = []
		cm_loss2 = 0.0

		### Determining Support Classes for EPABF-II
		mods = len(self.weights)
		if(lt_tilde[0] != 0.0):
			support_set2.append(labeltil[0])
			j = 1
			cm_loss2 += l_tilde[labeltil[j-1]]
			j = 1
			while(j != mods):
				if predicted_label in support_set2:
					if a == 0.0:
						if (l_tilde[predicted_label] != 0) and (l_tilde[labeltil[j]] != 0):
							support_set2.append(labeltil[j])
							cm_loss2 += l_tilde[labeltil[j]]
						j += 1
					else:
						temp1 = (l_tilde[predicted_label]/a) + ((((a*a*j)+1-a+(1/(modx*2*c)))/(a*a))*l_tilde[labeltil[j]])
						if(cm_loss2 < temp1):
							support_set2.append(labeltil[j])
							cm_loss2 += l_tilde[labeltil[j]]
							j += 1
						else:
							break
				else:
					if a == 0.0:
						if (l_tilde[labeltil[j]] != 0):
							support_set2.append(labeltil[j])
							cm_loss2 += l_tilde[labeltil[j]]
						j += 1
					else:
						temp1 = ((((a*a*j)+1-a+(1/(modx*2*c)))/(a*a))*l_tilde[labeltil[j]])
						if(cm_loss2 < temp1):
							support_set2.append(labeltil[j])
							cm_loss2 += l_tilde[labeltil[j]]
							j += 1
						else:
							break


		update_matrix = self.get_update_matrix(feature_vectors, calculated_label2, predicted_label, true_label, probabilities,support_set2,l_tilde,labeltil)
		self.update_weights(update_matrix)

class BPA:

	def __init__(self):
		self.gamma = 0.014
		self.dict_length = 100
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0

	def init_weights(self):
		weights = []
		for i in range(0,len(SYNSEP_CATEGORY_MAPPING)):
			weights.append([0.0] * self.dict_length)
		return weights

	def predict_label(self, feature_vectors):
		max = -99999999.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]
			# print("For label %s , we have w_i*x as %s" %(str(i),str(total)))
			if total >= max:
				max = total
				label = i
		return label

	def calc_probabilities(self, calculated_label):
		probabilities = [0.0] * len(self.weights)
		for i in range(0,len(probabilities)):
			probabilities[i] = self.gamma/len(self.weights)
			if i == calculated_label:
				probabilities[i] += (1 - self.gamma)
		return probabilities

	def random_sample(self, probabilities):
		number = random.random() * sum(probabilities)
		for i in range(0,len(probabilities)):
			if number < probabilities[i]:
				return i
				break
			number -= probabilities[i]
		return len(probabilities)-1

	def update_weights(self, update_matrix):
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]

	def get_update_matrix(self, feature_vectors, predicted_label,total, losst,a):
		update_matrix = self.init_weights()
		modxsq = 0.0
		
		### Calculating ||X||^2
		for j in range(0,len(feature_vectors)):
			modxsq += feature_vectors[j]*feature_vectors[j]

		for i in range(0,len(update_matrix)):
			if i == predicted_label:
				for j in range(0,len(feature_vectors)):
					update_matrix[predicted_label][j] = feature_vectors[j]*((((2*a)-1)*losst)/modxsq)

		return update_matrix

	def run(self, feature_vectors, true_label):
		self.number_of_rounds += 1.0
		calculated_label = self.predict_label(feature_vectors)
		probabilities = self.calc_probabilities(calculated_label)
		predicted_label = self.random_sample(probabilities)
		
		if true_label == predicted_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		

		if true_label == predicted_label:
			a = 1
		else:
			a = 0

		total = 0.0
		losst = 0
		
		for eachVector in range(0,len(feature_vectors)):
			total += feature_vectors[eachVector]*self.weights[predicted_label][eachVector]
		
		losst = max(0, (1+((1-2*a)*total)))
		update_matrix = self.get_update_matrix(feature_vectors, predicted_label,total, losst, a)
		
		self.update_weights(update_matrix)

class PA:
	def __init__(self):
		self.dict_length = 100
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0

	def init_weights(self):
		weights = []
		for i in range(0,len(SYNSEP_CATEGORY_MAPPING)):
			weights.append([0.0] * self.dict_length)
		return weights

	def predict_label(self, feature_vectors):
		max = -99999999.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]
			# print("For label %s , we have w_i*x as %s" %(str(i),str(total)))
			if total >= max:
				max = total
				label = i
		return label

	def update_weights(self, update_matrix):
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]

	def get_update_matrix(self, feature_vectors, calculated_label, predicted_label, true_label, support_set,l_tilde,labeltil):
		update_matrix = self.init_weights()
		step_lambda = [0.0 for x in range(0,len(update_matrix))] ## Lambdas for each label
		
		total_support_class_loss = 0.0
		
		modxsq = 0.0
		sum_step_lambda = 0.0
		
		for i in range(0,len(support_set)):
			total_support_class_loss += l_tilde[support_set[i]]

		### Calculating ||X||^2
		for j in range(0,len(feature_vectors)):
			modxsq += feature_vectors[j]*feature_vectors[j]

		### Determining lambdas for each support class label, non support class has lambda = 0
		for r in range(0,len(support_set)):
			step_lambda[support_set[r]] = (l_tilde[support_set[r]] - ((1/(1+(len(support_set))))*total_support_class_loss))/modxsq

		### Calculating the sum of all lambdas
		for j in range(0,len(step_lambda)):
			sum_step_lambda += step_lambda[j]

		sum_step_lambda -= step_lambda[true_label]

		### Determining the weight update corresponding to each label
		for i in range(0,len(update_matrix)):
			### if i is not predicted label(y_tilde), weight update:
			### w(i) = w_t(i) - lambda_i*(x_t)
			if i != true_label:
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = -feature_vectors[j] * step_lambda[i]
					
			### if i is predicted label(y_tilde), weight update:
			### w(y_tilde) = w_t(y_tilde) - sum(lambdas)*(x_t)*(1[y_tilde = y_t]/prob(y_tilde)-1)
			else:                
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = feature_vectors[j] * (sum_step_lambda)

		return update_matrix

	def run(self, feature_vectors, true_label):
		self.number_of_rounds += 1.0
		calculated_label = self.predict_label(feature_vectors)
		predicted_label = calculated_label
		
		if true_label == predicted_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		

		### Calculating l_tilde for each label
		l_tilde = [-1.0 for x in range(0,len(self.weights))]
		lt_tilde = [-1.0 for x in range(0,len(self.weights))]
		labeltil = [-1 for x in range(0,len(self.weights))]
		loss_tr = 0.0

		for eachVector in range(0,len(feature_vectors)):
			loss_tr += feature_vectors[eachVector]*self.weights[true_label][eachVector]
		

		# loss_r is (w_r)*x
		for r in range(0,len(self.weights)):
			if r != true_label:
				loss_r = 0.0
				for eachVector in range(0,len(feature_vectors)):
					loss_r += feature_vectors[eachVector]*self.weights[r][eachVector]
				
				l_tilde[r] = max(0.0, 1 - loss_tr + loss_r)
				lt_tilde[r] = max(0.0, 1 - loss_tr + loss_r)
				labeltil[r] = r
		
		modx = 0.0
		for j in range(0,len(feature_vectors)):
			modx += feature_vectors[j]*feature_vectors[j]

		### Sorting in Decreasing order the l_tildex and corresponding label
		for i in range(len(self.weights)):
			for j in range(0, len(self.weights)-i-1):
				if lt_tilde[j] < lt_tilde[j+1]:
					lt_tilde[j], lt_tilde[j+1] = lt_tilde[j+1], lt_tilde[j]
					labeltil[j], labeltil[j+1] = labeltil[j+1], labeltil[j]

		### If predicted label equals the true label, then a = 1/probability[predicted_label]
		support_set = []
		
		cm_loss = 0.0 # cumulative loss
		

		### Determining Support Classes for PA
		mods = len(self.weights)
		
		if(lt_tilde[0] != 0.0):
			support_set.append(labeltil[0])
			j = 1
			cm_loss += l_tilde[labeltil[j-1]]
			while(j!= mods):
				rhs = ((j+1)*l_tilde[labeltil[j]])
				if(cm_loss < rhs):
					support_set.append(labeltil[j])
					cm_loss += l_tilde[labeltil[j]]
					j += 1
				else:
					break

		update_matrix = self.get_update_matrix(feature_vectors, calculated_label, predicted_label, true_label,support_set,l_tilde,labeltil)
		self.update_weights(update_matrix)


class PA1:
	def __init__(self):
		self.dict_length = 100
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0

	def init_weights(self):
		weights = []
		for i in range(0,len(SYNSEP_CATEGORY_MAPPING)):
			weights.append([0.0] * self.dict_length)
		return weights

	def predict_label(self, feature_vectors):
		max = -99999999.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]
			# print("For label %s , we have w_i*x as %s" %(str(i),str(total)))
			if total >= max:
				max = total
				label = i
		return label

	def update_weights(self, update_matrix):
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]

	def get_update_matrix(self, feature_vectors, calculated_label, predicted_label, true_label, support_set,l_tilde,labeltil,c=0.1):
		update_matrix = self.init_weights()
		step_lambda = [0.0 for x in range(0,len(update_matrix))] ## Lambdas for each label
		
		total_support_class_loss = 0.0
		
		modxsq = 0.0
		sum_step_lambda = 0.0
		
		for i in range(0,len(support_set)):
			total_support_class_loss += l_tilde[support_set[i]]

		### Calculating ||X||^2
		for j in range(0,len(feature_vectors)):
			modxsq += feature_vectors[j]*feature_vectors[j]

		### Determining lambdas for each support class label, non support class has lambda = 0
		tempmax1 = (total_support_class_loss/len(support_set)) + ((c*modxsq)/len(support_set))
		tempmax2 = (total_support_class_loss/(len(support_set)+1))

		for r in range(0,len(support_set)):
			step_lambda[support_set[r]] = (l_tilde[support_set[r]] - min(tempmax1,tempmax2))/modxsq

		### Calculating the sum of all lambdas
		for j in range(0,len(step_lambda)):
			sum_step_lambda += step_lambda[j]

		sum_step_lambda -= step_lambda[true_label]

		### Determining the weight update corresponding to each label
		for i in range(0,len(update_matrix)):
			### if i is not predicted label(y_tilde), weight update:
			### w(i) = w_t(i) - lambda_i*(x_t)
			if i != true_label:
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = -feature_vectors[j] * step_lambda[i]
					
			### if i is predicted label(y_tilde), weight update:
			### w(y_tilde) = w_t(y_tilde) - sum(lambdas)*(x_t)*(1[y_tilde = y_t]/prob(y_tilde)-1)
			else:                
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = feature_vectors[j] * (sum_step_lambda)

		return update_matrix

	def run(self, feature_vectors, true_label,c=0.1):
		self.number_of_rounds += 1.0
		calculated_label = self.predict_label(feature_vectors)
		predicted_label = calculated_label
		
		if true_label == predicted_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		

		### Calculating l_tilde for each label
		l_tilde = [-1.0 for x in range(0,len(self.weights))]
		lt_tilde = [-1.0 for x in range(0,len(self.weights))]
		labeltil = [-1 for x in range(0,len(self.weights))]
		loss_tr = 0.0

		for eachVector in range(0,len(feature_vectors)):
			loss_tr += feature_vectors[eachVector]*self.weights[true_label][eachVector]
		

		# loss_r is (w_r)*x
		for r in range(0,len(self.weights)):
			if r != true_label:
				loss_r = 0.0
				for eachVector in range(0,len(feature_vectors)):
					loss_r += feature_vectors[eachVector]*self.weights[r][eachVector]
				
				l_tilde[r] = max(0.0, 1 - loss_tr + loss_r)
				lt_tilde[r] = max(0.0, 1 - loss_tr + loss_r)
				labeltil[r] = r
		
		modx = 0.0
		for j in range(0,len(feature_vectors)):
			modx += feature_vectors[j]*feature_vectors[j]
		
		### Sorting in Decreasing order the l_tildex and corresponding label
		for i in range(len(self.weights)):
			for j in range(0, len(self.weights)-i-1):
				if lt_tilde[j] < lt_tilde[j+1]:
					lt_tilde[j], lt_tilde[j+1] = lt_tilde[j+1], lt_tilde[j]
					labeltil[j], labeltil[j+1] = labeltil[j+1], labeltil[j]

		### If predicted label equals the true label, then a = 1/probability[predicted_label]
		support_set = []
		
		cm_loss = 0.0 # cumulative loss

		### Determining Support Classes for PA-I
		mods = len(self.weights)
		
		if(lt_tilde[0] != 0.0):
			support_set.append(labeltil[0])
			j = 1
			cm_loss += l_tilde[labeltil[j-1]]
			while(j!= mods):
				temp1 = ((j+1)*l_tilde[labeltil[j]])
				temp2 = (((j+2)*c)/modx) - l_tilde[labeltil[j]]
				rhs = min(temp1,temp2)
				if(cm_loss < rhs):
					support_set.append(labeltil[j])
					cm_loss += l_tilde[labeltil[j]]
					j += 1
				else:
					break

		update_matrix = self.get_update_matrix(feature_vectors, calculated_label, predicted_label, true_label, support_set,l_tilde,labeltil)
		self.update_weights(update_matrix)


class PA2:
	def __init__(self):
		self.dict_length = 100
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0

	def init_weights(self):
		weights = []
		for i in range(0,len(SYNSEP_CATEGORY_MAPPING)):
			weights.append([0.0] * self.dict_length)
		return weights

	def predict_label(self, feature_vectors):
		max = -99999999.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]
			if total >= max:
				max = total
				label = i
		return label

	def update_weights(self, update_matrix):
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]

	def get_update_matrix(self, feature_vectors, calculated_label, predicted_label, true_label, support_set,l_tilde,labeltil,c=0.1):
		update_matrix = self.init_weights()
		step_lambda = [0.0 for x in range(0,len(update_matrix))] ## Lambdas for each label
		
		total_support_class_loss = 0.0
		
		modxsq = 0.0
		sum_step_lambda = 0.0
		
		for i in range(0,len(support_set)):
			total_support_class_loss += l_tilde[support_set[i]]

		### Calculating ||X||^2
		for j in range(0,len(feature_vectors)):
			modxsq += feature_vectors[j]*feature_vectors[j]

		### Determining lambdas for each support class label, non support class has lambda = 0
		temp1 = (modxsq + (1/(2*c)))
		temp2 = ((len(support_set)+1)*modxsq) + (len(support_set)/(2*c))
		for r in range(0,len(support_set)):
			step_lambda[support_set[r]] = (l_tilde[support_set[r]] - ((temp1/temp2)*total_support_class_loss))/modxsq

		### Calculating the sum of all lambdas
		for j in range(0,len(step_lambda)):
			sum_step_lambda += step_lambda[j]

		sum_step_lambda -= step_lambda[true_label]

		### Determining the weight update corresponding to each label
		for i in range(0,len(update_matrix)):
			### if i is not predicted label(y_tilde), weight update:
			### w(i) = w_t(i) - lambda_i*(x_t)
			if i != true_label:
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = -feature_vectors[j] * step_lambda[i]
					
			### if i is predicted label(y_tilde), weight update:
			### w(y_tilde) = w_t(y_tilde) - sum(lambdas)*(x_t)*(1[y_tilde = y_t]/prob(y_tilde)-1)
			else:                
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = feature_vectors[j] * (sum_step_lambda)

		return update_matrix

	def run(self, feature_vectors, true_label,c=0.1):
		self.number_of_rounds += 1.0
		calculated_label = self.predict_label(feature_vectors)
		predicted_label = calculated_label
		
		if true_label == predicted_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		

		### Calculating l_tilde for each label
		l_tilde = [-1.0 for x in range(0,len(self.weights))]
		lt_tilde = [-1.0 for x in range(0,len(self.weights))]
		labeltil = [-1 for x in range(0,len(self.weights))]
		loss_tr = 0.0

		for eachVector in range(0,len(feature_vectors)):
			loss_tr += feature_vectors[eachVector]*self.weights[true_label][eachVector]
		

		# loss_r is (w_r)*x
		for r in range(0,len(self.weights)):
			if r != true_label:
				loss_r = 0.0
				for eachVector in range(0,len(feature_vectors)):
					loss_r += feature_vectors[eachVector]*self.weights[r][eachVector]
				
				l_tilde[r] = max(0.0, 1 - loss_tr + loss_r)
				lt_tilde[r] = max(0.0, 1 - loss_tr + loss_r)
				labeltil[r] = r
		
		modx = 0.0
		for j in range(0,len(feature_vectors)):
			modx += feature_vectors[j]*feature_vectors[j]

		### Sorting in Decreasing order the l_tilde copy and corresponding label
		for i in range(len(self.weights)):
			for j in range(0, len(self.weights)-i-1):
				if lt_tilde[j] < lt_tilde[j+1]:
					lt_tilde[j], lt_tilde[j+1] = lt_tilde[j+1], lt_tilde[j]
					labeltil[j], labeltil[j+1] = labeltil[j+1], labeltil[j]

		### If predicted label equals the true label, then a = 1/probability[predicted_label]
		support_set = []
		
		cm_loss = 0.0 # cumulative loss
		

		### Determining Support Classes for PA
		mods = len(self.weights)
		
		if(lt_tilde[0] != 0.0):
			support_set.append(labeltil[0])
			j = 1
			cm_loss += l_tilde[labeltil[j-1]]
			while(j!= mods):
				temp1 = ((j+1)*modx) + ((j)/(2*c))
				temp2 = (modx) + (1/(2*c))
				rhs = (temp1/temp2)*l_tilde[labeltil[j]]
				if(cm_loss < rhs):
					support_set.append(labeltil[j])
					cm_loss += l_tilde[labeltil[j]]
					j += 1
				else:
					break

		update_matrix = self.get_update_matrix(feature_vectors, calculated_label, predicted_label, true_label, support_set,l_tilde,labeltil)
		self.update_weights(update_matrix)



def generateSynsep():
	feature_vector = []
	no_of_features = 100

	for i in range(0,no_of_features):
		ran = random.uniform(-1,1)
		feature_vector.append(ran)

	maxi = -999999999
	true_label = 0
	for i in range(0, len(SYNSEP_CATEGORY_MAPPING)):
		u1 = [1]*no_of_features
		for j in range(0, len(SYNSEP_CATEGORY_MAPPING)):
			u1[i+10*j] = -1
		for k in range(0,no_of_features):
			temp = u1[k]*feature_vector[k]
		if temp >= maxi:
			maxi = temp
			true_label = i
	
	return feature_vector, true_label




def main():
	erph = [0.0 for x in range(0,1000)]
	erb = [0.0 for x in range(0,1000)]
	er = [0.0 for x in range(0,1000)]
	er1 = [0.0 for x in range(0,1000)]
	er2 = [0.0 for x in range(0,1000)]
	erpa = [0.0 for x in range(0,1000)]
	erpa1 = [0.0 for x in range(0,1000)]
	erpa2 = [0.0 for x in range(0,1000)]

	rrph = [0 for x in range(0,1000)]
	
	iterations = 10
	for k in range(0, iterations):
		print("Epoch: %d" %int(k))
		banditron = Banditron()
		epabf = EPABF()
		epabf1 = EPABF1()
		epabf2 = EPABF2()
		bpa = BPA()
		pa = PA()
		pa1 = PA1()
		pa2 = PA2()

		error_list = list()
		error_list = list()
		error_list1 = list()
		error_list2 = list()
		error_listph = list()
		error_listpa = list()
		error_listpa1 = list()
		error_listpa2 = list()

		roundsph = list()

		for t in range(0,100000):
			feature_vectors, true_label = generateSynsep()
			banditron.run(feature_vectors, true_label)
			epabf.run(feature_vectors, true_label)
			epabf1.run(feature_vectors, true_label)
			epabf2.run(feature_vectors, true_label)
			
			pa.run(feature_vectors, true_label)
			pa1.run(feature_vectors, true_label)
			pa2.run(feature_vectors, true_label)
			bpa.run(feature_vectors, true_label)


			if ((t+1)%100) == 0:				
				print("%s rounds completed with error rate %s by Banditron" %(str(t+1),str(banditron.error_rate)))
				print("%s rounds completed with error rate %s by EPABF" %(str(t+1),str(epabf.error_rate)))
				print("%s rounds completed with error rate %s by EPABF-I" %(str(t+1),str(epabf1.error_rate)))
				print("%s rounds completed with error rate %s by EPABF-II" %(str(t+1),str(epabf2.error_rate)))
				print("%s rounds completed with error rate %s by BPA" %(str(t+1),str(bpa.error_rate)))
				print("%s rounds completed with error rate %s by PA" %(str(t+1),str(pa.error_rate)))
				print("%s rounds completed with error rate %s by PA-I" %(str(t+1),str(pa1.error_rate)))
				print("%s rounds completed with error rate %s by PA-II" %(str(t+1),str(pa2.error_rate)))
				
				roundsph.append(bpa.number_of_rounds)

				error_listb.append(banditron.error_rate)
				error_list.append(epabf.error_rate)
				error_list1.append(epabf1.error_rate)
				error_list2.append(epabf2.error_rate)
				error_listpa.append(pa.error_rate)
				error_listpa1.append(pa1.error_rate)
				error_listpa2.append(pa2.error_rate)				
				error_listph.append(bpa.error_rate)

				print("=================================")

		for i in range(0,1000):
			erb[i] += error_listb[i]
			er[i] += error_list[i]
			er1[i] += error_list1[i]
			er2[i] += error_list2[i]
			erpa[i] += error_listpa[i]
			erpa1[i] += error_listpa1[i]
			erpa2[i] += error_listpa2[i]
			erph[i] += error_listph[i]

	for i in range(0,1000):		
		rrph[i] = 100*(i+1)

	for i in range(0,1000):
		erb[i] = erb[i]/iterations
		er[i] = er[i]/iterations
		er1[i] = er1[i]/iterations
		er2[i] = er2[i]/iterations
		erpa[i] = erpa[i]/iterations
		erpa1[i] = erpa1[i]/iterations
		erpa2[i] = erpa2[i]/iterations		
		erph[i] = erph[i]/iterations



	a = {'error_rate_ph': erph, 'error_rate_b': erb, 'error_rate_epabf': er, 'error_rate_epabf1': er1, 'error_rate_epabf2': er2, 'error_rate_pa': erpa, 'error_rate_pa1': erpa1, 'error_rate_pa2': erpa2, 'rounds': rrph}
	# a = {'error_rate_epabf2': er2, 'rounds': rrph}

	datavalall = open('abc.txt','w+')
	datavalall.write(str(a))
	datavalall.close()


if __name__=="__main__":
	main()