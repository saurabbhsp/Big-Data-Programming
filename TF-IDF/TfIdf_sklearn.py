import os
import sys
import numpy as np
import re
import math
import json

from mpi4py import MPI


"""This method list of all inputs.
Should be changed as per the input format"""

def buildInput(basePath):
	"""Generate list of all possible files in directory 
	and subdirectory, with depth 1"""
	pathList = []
	for dir in os.listdir(basePath):
		for file in os.listdir(os.path.join(basePath, dir)):
			pathList.append(os.path.join(basePath, dir, file))
	return pathList

def readFile(path):
	with open(path, 'r', errors = 'surrogateescape') as inputFile:
		print("Executing for "+inputFile.name)
		text = inputFile.read()
	return text

def writeToFile(path, text):
	with open(path, 'w') as outputFile:
		outputFile.write(text)

def cleanText(text):


	"""Convert text to lower case"""
	text = text.lower()
	"""Remove special characters + email addresses + alpha numeric entries"""
	text = re.sub(r'\S*@\S*\s?|([^\s\w]|_)+|\w*\d\w*|[^A-Za-z0-9\s]|^\d+\s|\s\d+\s|\s\d+$', '', text)
	"""remove new lines"""
	text = text.replace("\n", " ")
	"""Replace more than one tabs with space"""
	text = re.sub('\t+',' ', text)
	"""Finally remove more than one spaces with space"""
	text = re.sub(' +',' ', text)
	return text







"""read input path as command line parameter"""
inputPath = sys.argv[1]
outputPath = sys.argv[2]

if not os.path.exists(outputPath):
	os.makedirs(outputPath)

if not os.path.exists(os.path.join(outputPath, "TF")):
	os.makedirs(os.path.join(outputPath, "TF"))

if not os.path.exists(os.path.join(outputPath, "TFIDF")):
	os.makedirs(os.path.join(outputPath, "TFIDF"))

pathList = buildInput(inputPath)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = "english", ngram_range = (1, 3))

textList = []
for file in pathList:
	text = readFile(file)
	textList.append(cleanText(text))

resultTF = vectorizer.fit_transform(textList)

print(json.dumps(resultTF[0]))