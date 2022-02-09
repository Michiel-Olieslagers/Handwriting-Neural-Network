import numpy as np
import random
import tensorflow as tf
import mnist_loader
import time
from PIL import Image


class Network(object):
	def __init__(self,sizes):
		self.numNeurons = len(sizes)
		self.sizes = sizes
		checkCount = 0
		for x,y in zip(sizes[:-1],sizes[1:]):
			checkCount = checkCount + y + y*x
		with open('values.txt', 'r') as f:
			lines = f.readlines()
		if lines != []:
			ls = lines[0].split(",")
			ls = ls[:-1]
			ls = np.array(ls)
			ls = ls.astype(np.float)
		else:
			ls = []
		if len(ls) != checkCount:
			self.bias = [np.random.randn(y,1) for y in sizes[1:]]
			self.weight = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
		else:
			totalCount = 0
			ls1 = []
			ls2 = []
			for y in sizes[1:]:
				ls2 = []
				for i in range(0,y):
					ls2.append(np.array([ls[totalCount]]))
					totalCount += 1
				ls1.append(np.array(ls2))
			self.bias = ls1
			ls1 = []
			ls2 = []
			ls3 = []
			for x,y in zip(sizes[:-1],sizes[1:]):
				ls2 = []
				for i in range(0,y):
					ls3 = []
					for j in range(0,x):
						ls3.append(ls[totalCount])
						totalCount += 1
					ls2.append(np.array(ls3))
				ls1.append(np.array(ls2))
			self.weight = ls1

	def feedForward(self,a):
		for b,w in zip(self.bias,self.weight):
			a = sigmoid(np.dot(w,a) + b)
		return a

	def stochasticGradientDescent(self,trainingData, epochs, miniBatchSize, learningRate, testData=None):
		if testData: nTest = len(testData)
		n = len(trainingData)
		inp = input("Train network?[Y/N]")
		if inp == "Y":
			for j in range(epochs):
				random.shuffle(trainingData)
				miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0,n,miniBatchSize)]
				for miniBatch in miniBatches:
					self.updateMiniBatch(miniBatch,learningRate)
			with open('values.txt','w') as f:
				for i in self.bias:
					for j in i:
						for k in j:
							f.write("{:.9f},".format(k))
				for i in self.weight:
					for j in i:
						for k in j:
							f.write("{:.9f},".format(k))
		if testData:
			self.evaluateOneByOne(testData)
			print ("Total: {0} / {1}".format(self.evaluate(testData), nTest))
		else:
			print ("Complete")

	def updateMiniBatch(self, miniBatch, learningRate):
		nablaB = [np.zeros(b.shape) for b in self.bias]
		nablaW = [np.zeros(w.shape) for w in self.weight]
		for x,y in miniBatch:
			deltaNablaB, deltaNablaW = self.backProp(x,y)
			nablaB = [nb+dnb for nb, dnb in zip(nablaB, deltaNablaB)]
			nablaW = [nw+dnw for nw, dnw in zip(nablaW, deltaNablaW)]
		self.weight = [w-(learningRate/len(miniBatch)) * nw for w,nw in zip(self.weight, nablaW)]
		self.bias = [b-(learningRate/len(miniBatch)) * nb for b,nb in zip(self.bias,nablaB)]

	def backProp(self, x, y):
		nablaB = [np.zeros(b.shape) for b in self.bias]
		nablaW = [np.zeros(w.shape) for w in self.weight]
		activation = x
		activations = [x]
		zs = []
		for b, w in zip(self.bias,self.weight):
			z = np.dot(w,activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1])
		nablaB[-1] = delta
		nablaW[-1] = np.dot(delta, activations[-2].transpose())
		for l in range(2, self.numNeurons):
			z = zs[-l]
			sp = sigmoidPrime(z)
			delta = np.dot(self.weight[-l+1].transpose(), delta) * sp
			nablaB[-l] = delta
			nablaW[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nablaB, nablaW)

	def evaluate(self,testData):
		testResults = [(np.argmax(self.feedForward(x)), y) for (x,y) in testData]
		return sum(int(x==y) for (x,y) in testResults)

	def evaluateOneByOne(self,testData):
		for (x,y) in testData:
			mat = np.reshape(x,(28,28))
			img = Image.fromarray(np.uint8(mat * 255) , 'L')
			img.resize((256,256))
			img.show()
			testResults = self.feedForward(x)
			total = sum(testResults)
			for i in range(0,len(testResults)):
				print("{0} : {1}%".format(i,(testResults[i]/total*100)))
			input()

	def costDerivative(self, outputActivations, y):
		return (outputActivations-y)

def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

def sigmoidPrime(z):
	return sigmoid(z) * (1 - sigmoid(z))

trainingData, validationData, testData = mnist_loader.load_data_wrapper()
trainingData = list(trainingData)
testData = list(testData)
net = Network([784,30,10])
net.stochasticGradientDescent(trainingData,30,10,3,testData=testData)
