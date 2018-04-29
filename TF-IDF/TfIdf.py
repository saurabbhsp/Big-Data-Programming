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

	"""Following cleanup is specific to 20 newsgroup dataset"""
	text = re.sub(r'^Xref: .*\n?|^Path: .*\n?|^From: .*\n?|'+
	'^Newsgroups: .*\n?|^Subject: *\n?|^Summary: *\n?|^Keywords: *\n?|^Message-ID: *\n?|'+
	'^Date: *\n?|^Expires: *\n?|^Followup-To: *\n?|^Distribution: *\n?|^Organization: *\n?|'+
	'^Approved: *\n?|^Supersedes: *\n?|^Lines: *\n?|^Re: *\n?|^Reply-To: *\n?|'+
	'^Article-I.D.: *\n?|^References: *\n?|^NNTP-Posting-Host: *\n?', '', text, flags=re.MULTILINE)

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


def generateNormalizedTermFrequency(tokens, documentFrequency):
	termFrequency = {}
	tokenCount = len(tokens)
	for token in tokens:
		if token in termFrequency:
			"""Division by tokenCount in document will normalize the termfrequency"""
			termFrequency[token] = termFrequency[token] + 1.0/tokenCount
		else:
			termFrequency[token] = 1.0/tokenCount

			"""Update copy of local document frequency"""
			if token in documentFrequency:
				documentFrequency[token] = documentFrequency[token] + 1
			else:
				documentFrequency[token] = 1
	return termFrequency, documentFrequency

def generateIDF(documentFrequency, totalDocuments):
	inverseDocumentFrequency = {}
	for token in documentFrequency:
		inverseDocumentFrequency[token] = math.log((totalDocuments * 1.0)/documentFrequency[token])
	return inverseDocumentFrequency

def generateTfIdf(termFrequency, inverseDocumentFrequency, metaData):
	tfidf = {}

	for token in termFrequency:
		tfidf[metaData["dictionary"][token]] = termFrequency[token] * inverseDocumentFrequency[token]
	return tfidf

def wordgramGenerator(tokens, n, stopWords = []):
	ngrams = []
	for index in range(0, len(tokens)):
		candidate = tokens[index:index + n]
		if candidate[0] in stopWords and candidate[-1] in stopWords:
			continue
		ngram = ' '.join(candidate)
		ngrams.append(ngram)
	return ngrams

def ngramGenerator(text, ngramRange = (1, 1), stopWords = []):

	tokens = text.split(" ")
	ngrams = []
	for i in range(ngramRange[0], ngramRange[1]+1):
		ngrams.extend(wordgramGenerator(tokens, i, stopWords))
	return ngrams



"""MPI variables initialization"""
comm = MPI.COMM_WORLD
processorCount = comm.Get_size()
currentRank = comm.Get_rank()
root = 0

"""read input path as command line parameter"""
inputPath = sys.argv[1]
outputPath = sys.argv[2]
ngramS = int(sys.argv[3])
ngramE = int(sys.argv[4])

taskChunks = None
pathList = None

termFrequencies = []
documentFrequency = {}
inverseDocumentFrequency = None
metaData = None

"""Only root will execute the following code to generate possible inputs paths.
These input paths will be stored on a list. This list is
further broken into smaller parts 
so that it can be passed on to workers"""
if currentRank == root:
	startTime = MPI.Wtime()
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)

	if not os.path.exists(os.path.join(outputPath, "TF")):
		os.makedirs(os.path.join(outputPath, "TF"))

	if not os.path.exists(os.path.join(outputPath, "TFIDF")):
		os.makedirs(os.path.join(outputPath, "TFIDF"))

	pathList = buildInput(inputPath)
	taskChunks = np.array_split(pathList, processorCount)


"""Every worker will have a copy of stopwords.
These stopwords are loaded from a file.
Source for stopwords: https://github.com/saurabbhsp/stopwords"""
with open('stopwords', 'r') as inputFile:
	stopWords = inputFile.readlines()

stopWords = [word.strip().lower() for word in stopWords]
stopWords.append("")

"""Divide task among workers"""
inputFiles = comm.scatter(taskChunks, root)


"""Calculate term frequency and the relative document frequency
Here first I am reading text from the input file.
Followed by clean up of text file. The cleanText procedure will remove 
characters. This data cleanup script is written for news20 data set and 
hence there are some methods specific to it."""
for file in inputFiles:
	text = readFile(file)
	text = cleanText(text)
	tokens = ngramGenerator(text, (ngramS, ngramE), stopWords)
	tokenTermFrequency, documentFrequency = generateNormalizedTermFrequency(tokens, documentFrequency)
	writeToFile(os.path.join(outputPath, "TF",os.path.basename(file)+".json"), json.dumps(tokenTermFrequency, indent=2))
	termFrequencies.append(tokenTermFrequency)


"""Gather all the relative document frequency"""
relativeDocumentFrequencies = comm.gather(documentFrequency, root)
if currentRank == root:

	documentFrequency = {}
	metaData = {}
	metaData["dictionary"] = {}
	vectorCount = 0
	totalDocuments = len(pathList)

	for relativeDocumentFrequency in relativeDocumentFrequencies:
		for token in relativeDocumentFrequency:
			if token in documentFrequency:
				documentFrequency[token] = documentFrequency[token] + relativeDocumentFrequency[token]
			else:
				documentFrequency[token] = relativeDocumentFrequency[token]
				metaData["dictionary"][token] = vectorCount
				vectorCount+=1
	metaData["vocabLength"] = vectorCount
	inverseDocumentFrequency = generateIDF(documentFrequency, totalDocuments)
	writeToFile(os.path.join(outputPath, "metadata.json"), json.dumps(metaData, indent=2))
	writeToFile(os.path.join(outputPath, "IDF.json"), json.dumps(inverseDocumentFrequency, indent=2))

"""Broadcast the idf to every node"""
inverseDocumentFrequency = comm.bcast(inverseDocumentFrequency, root)
metaData = comm.bcast(metaData, root)


"""Calculate tfidf"""
for termFrequency, file in zip(termFrequencies, inputFiles):
	tfidf = generateTfIdf(termFrequency, inverseDocumentFrequency, metaData)
	writeToFile(os.path.join(outputPath, "TFIDF",os.path.basename(file)+".json"), json.dumps(tfidf, indent=2))


if currentRank == root:
	endTime = MPI.Wtime()
	print("Total time taken " + str((endTime-startTime)/60))