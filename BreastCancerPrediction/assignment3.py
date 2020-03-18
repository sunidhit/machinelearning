import numpy as np

class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k
		self.X1 = []
		self.y1 = []

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def most_frequent_neighbor(self,neighbors):
		k_neighbors=neighbors[:self.k]
		most_frequent_label=np.bincount(k_neighbors[1]).argmax()
		return most_frequent_label


	def train(self, X, y):
		#training logic here
		#input is an array of features and labels
		#memorizing data
		self.X1 = X
		self.y1 = y
		None

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		num_of_rows,num_of_columns=X.shape
		predictedlabels=np.empty(shape=[0,num_of_rows])
		train_rows,train_columns = self.X1.shape
		for i in range(num_of_rows):
			distance_array = []
			for j in range(train_rows):
				distance_array.append((self.distance(X[i], self.X1[j]), self.y1[j]))
			distance_array = sorted(distance_array, key=lambda x: (x[0]))
			predictedlabels=np.append(predictedlabels,self.most_frequent_neighbor(distance_array))
		return predictedlabels

class ID3:
	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range
		#self.tree = None

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)
		nr,nc = X.shape
		feature=[]
		for i in range(nc):
			feature.append(i)
		self.initial_data=y
		tree = self.create_tree(categorical_data,y,feature)
		self.tree = tree

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		predicted_labels=[]
		no_of_rows,no_of_columns = X.shape
		query={}
		for i in categorical_data:
			key = range(len(i))
			query = dict(zip(key,i))
			predicted_labels.append(self.traverse(query,self.tree))
		return np.ravel(predicted_labels)

	def traverse(self, query,tree):
		for i in list(query.keys()):
			if i in list(tree.keys()):
				try:
					result = self.tree[i][query[i]]
				except:
					return 1
				if isinstance(result,dict):
					return self.traverse(query,result)
				else:
					return result


	def initial_entropy(self, y):
		values, counts = np.unique(y, return_counts=True)
		entropy = 0.0
		for i in range(values.max(0)):
			prob = float(counts[i] / np.sum(counts))
			entropy = entropy + (-prob * np.log(prob))
		return entropy

	def sub_entropy(self, y):
		values, counts = np.unique(y, return_counts=True)
		entropy = 0
		for i in range(values.size):
			prob = counts[i] / np.sum(counts)
			entropy = entropy + (-prob * np.log(prob))
		return entropy

	def gain(self, X, y, split_on):
		root_entropy = self.initial_entropy(y)
		feature_matrix = X[:, split_on]
		gain = 0.0
		entropy = 0.0
		values, counts = np.unique(feature_matrix, return_counts=True)
		for i in range(values.size):
			r = np.where(feature_matrix == values[i])
			split_ratio = float(counts[i] / np.sum(counts))
			c_entropy = split_ratio * self.sub_entropy(np.take(y, r))
			entropy = entropy + c_entropy
		return root_entropy - entropy

	def best_split(self, X, y,feature):
		best_gain = []
		for i in feature:
			best_gain.append(self.gain(X, y, i))
		return np.argmax(best_gain)

	def create_tree(self, X, y,feature):
		new_feature = []
		leafnode = np.unique(self.initial_data)[np.argmax(np.unique(self.initial_data, return_counts=True)[1])]
		v, c = np.unique(y, return_counts=True)
		leny = len(np.unique(y))
		if leny == 1:
			return y[0]
		elif (len(feature) == 1):
			return leafnode
		elif (len(X) <= 1):
			return self.initial_data[np.argmax(np.unique(self.initial_data, return_counts=True)[1])]
		else:
			best_split_index = self.best_split(X, y, feature)
			best_split_index1 =feature[best_split_index]
			tree = {best_split_index: {}}
			unique_values, count = np.unique(X[:, best_split_index1], return_counts=True)
			for j in feature:
				if (j != best_split_index1):
					new_feature.append(j)
			for i in unique_values:
				new_data = X[X[:, best_split_index1] == i]
				y_new = y[X[:, best_split_index1] == i]
				tree[best_split_index][i] = self.create_tree(new_data, y_new, new_feature)
		return tree




class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		for steps in range(steps):
			i = steps % y.size
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)
			label = self.step_function(xi)
			if label==0 and y[i]==1:
					self.w = self.w + (self.lr * xi)
					self.b = self.b + self.lr
			elif label==1 and y[i]==0:
					self.w = self.w - (self.lr * xi)
					self.b = self.b - self.lr
		return None

	def step_function(self,x):
		predicted_label = 0
		step_value = np.multiply(self.w,x) + self.b
		if np.sum(step_value) > 0:
			predicted_label = 1
		else:
			predicted_label = 0
		return predicted_label


	def predict(self, X):
		#Run model here
		num_of_rows, num_of_columns = X.shape
		predicted_labels = np.empty(shape=[0, num_of_rows])
		for i in range(num_of_rows):
			predicted_labels=np.append(predicted_labels,self.step_function(X[i]))
		return predicted_labels



class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)


			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi) 
			#print("loss is:",loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		print("predicting now")
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b
		self.x = 0

	def forward(self, input):
		#Write forward pass here
		 #self.x = input
		self.x = input
		return input.dot(self.w) + self.b

	def backward(self, gradients):
		#Write backward pass here
		w_dash = np.dot(np.transpose(self.x), gradients)
		x_dash = np.dot(gradients, np.transpose(self.w))
		self.w = self.w - (self.lr * w_dash)
		self.b = self.b - (self.lr * gradients)
		return x_dash

class Sigmoid:

	def __init__(self):
		self.t = 0
		None

	def forward(self, input):
		#Write forward pass here
		self. t = 1 / (1 + np.exp(-input))
		return 1 / (1 + np.exp(-input))

	def backward(self, gradients):
		#Write backward pass here
		# derivative of t
		sig = self.t * (1 - self.t)
		return sig*gradients