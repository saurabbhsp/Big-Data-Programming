# 1 Distributed TF-IDF
This project provides a method calculate TF-IDF on distributed environment using MPI message passing interface. There are two methods to approach this problem and both have their own shortcomings and advantages.
1) All processors will try to calculate term frequency for same document. This approach requires methods to split the text on all worker nodes. Then each worker node will try to calculate term frequency for the chunk of the document available to it. Then one or more workers combines the term frequency resulting into term frequency of the initial document. This methods results into added communication overhead among worker nodes and may result into poor performance. However this method is ideal while dealing with large text files.
2) In second approach each worker will try to calculate term frequency for different documents. Each worker should have access to the shared data. An alternative to this will include having a root processor communicating raw text to other workers which will again result into communication overhead similar to first approach.

# 2 Problem setting
This project will use python and open MPI for message passing to calculate TF-IDF vectors on distributed environment. Of the above mentioned approaches this project will use the second one. Each processor will calculate the term frequency for different documents. 

## 2.1 Assumptions
For this problem it is assumed that all workers have access to DFS (Distributed file system) and reading and writing of output is occurring on DFS. This means that all the workers will have access to all the data on the file system. There are other means to make data available on worker nodes like usage of document database but that will require some change in data reading module of this project.

## 2.2 Data set and stopwords
This code has been verified with [Twenty News Group](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) data available at UCI machine learning repository. For cleanup I am using a self compiled stopwords  list available in my [repository.](https://github.com/saurabbhsp/stopwords)


# 3 Utility methods


```python
import os
import sys

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
```

The above block includes methods for File I/O. First method buildInput generates a list of all files found in the directory basePath. This methods seeks all the files recursively till depth of 1. Second method is readFile. This method reads the raw text from provided file. Finally writeToFile method is used to store text to the provided file.
# 4 Text cleanup
```python
import re

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
```

The above procedure cleans text for down stream processing. This clean up script is written for Twenty newsgroup dataset and may need changes as per usecase.
```
Xref: cantaloupe.srv.cs.cmu.edu alt.atheism:49960 alt.atheism.moderated:
713 news.answers:7054 alt.answers:126
Path: cantaloupe.srv.cs.cmu.edu!crabapple.srv.cs.cmu.edu!bb3.andrew.cmu.edu!news.sei.cmu.edu!
cis.ohio-state.edu!magnus.acs.ohio-state.edu!usenet.ins.cwru.edu!
agate!spool.mu.edu!uunet!pipex!ibmpcug!mantis!mathew
From: mathew <mathew@mantis.co.uk>
Newsgroups: alt.atheism,alt.atheism.moderated,news.answers,alt.answers
Subject: Alt.Atheism FAQ: Atheist Resources
Summary: Books, addresses, music -- anything related to atheism
Keywords: FAQ, atheism, books, music, fiction, addresses, contacts
Message-ID: <19930329115719@mantis.co.uk>
Date: Mon, 29 Mar 1993 11:57:19 GMT
Expires: Thu, 29 Apr 1993 11:57:19 GMT
Followup-To: alt.atheism
Distribution: world
Organization: Mantis Consultants, Cambridge. UK.
Approved: news-answers-request@mit.edu
Supersedes: <19930301143317@mantis.co.uk>
Lines: 290

Archive-name: atheism/resources
Alt-atheism-archive-name: resources
Last-modified: 11 December 1992
Version: 1.0

                              Atheist Resources

                      Addresses of Atheist Organizations

                                     USA

FREEDOM FROM RELIGION FOUNDATION

Darwin fish bumper stickers and assorted other atheist paraphernalia are
available from the Freedom From Religion Foundation in the US.

Write to:  FFRF, P.O. Box 750, Madison, WI 53701.
Telephone: (608) 256-8900

```
Above is a sample file(alt.athesim/49960) from dataset. The first regex cleans up unwanted markers from text file(line containing Distribution:, Lines: and so on) followed by clean up of special characters. For instance the first regex will remove following text from the input file.
```
Xref: cantaloupe.srv.cs.cmu.edu alt.atheism:49960 alt.atheism.moderated:
713 news.answers:7054 alt.answers:126
Path: cantaloupe.srv.cs.cmu.edu!crabapple.srv.cs.cmu.edu!bb3.andrew.cmu.edu!news.sei.cmu.edu!
cis.ohio-state.edu!magnus.acs.ohio-state.edu!
usenet.ins.cwru.edu!agate!spool.mu.edu!uunet!pipex!ibmpcug!mantis!mathew
From: mathew <mathew@mantis.co.uk>
Newsgroups: alt.atheism,alt.atheism.moderated,news.answers,alt.answers
Subject: Alt.Atheism FAQ: Atheist Resources
Summary: Books, addresses, music -- anything related to atheism
Keywords: FAQ, atheism, books, music, fiction, addresses, contacts
Message-ID: <19930329115719@mantis.co.uk>
Date: Mon, 29 Mar 1993 11:57:19 GMT
Expires: Thu, 29 Apr 1993 11:57:19 GMT
Followup-To: alt.atheism
Distribution: world
Organization: Mantis Consultants, Cambridge. UK.
Approved: news-answers-request@mit.edu
Supersedes: <19930301143317@mantis.co.uk>
Lines: 290

Archive-name: atheism/resources
Alt-atheism-archive-name: resources
Last-modified: 11 December 1992
Version: 1.0
```
This data is just metadata and hence removed during cleanup process. This is followed by regex for removal of new line character, tabs and extra spaces. Finally this method returns cleaned text.

# 5 Word tokenizer

```python
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
```
Once the text has been cleaned the next task is to generate tokens. There are two methods used for this task. First is wordgramGenerator. This methods expects list of tokens and list of stopwords and n gram count. This method essentially generates bigram trigrams or ngrams depending on input parameter n. Removing stopwords from a one gram token is easy however doing same for bigrams and trigrams or ngrams is difficult. 
```
The quick fox
the apple is
is the
```
Above are some of the possible n grams. Each one of them have some stopwords present however they also have some non stop words present. To remove stopwords from tokens, wordgramGenerator method checks for first and last word. If both of them are stopwords then the token is dropped. For instance in above case the first token "The quick fox" will be accepted however other two will be rejected. If stopwords list provided is empty then all word grams will be generated. The second method ngramGenerator accepts text, stopwords and ngramRange. This method uses wordgramGenerator to generate tokens for provided ngram range.

# 6 TF and DF generation
```python
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
```
The above method calculates both term frequency and document frequency. This method excepts two parameters. First one is the tokens and second one is existing documentFrequency dictionary. This document frequency dictionary is common for entire corpus. For term frequency a new dictionary is created withing this method. For text processing the vector is going to be sparse. Because of this a dictionary instead of list or a vector. There are two important blocks in the above method.
```python
if token in termFrequency:
	"""Division by tokenCount in document will normalize the termfrequency"""
	termFrequency[token] = termFrequency[token] + 1.0/tokenCount
else:
	termFrequency[token] = 1.0/tokenCount
```  
In the first block generateNormalizedTermFrequency procedure checks if the token is already present in the term frequency. If it is then the method updates the term frequency in dictionary by adding <b>1.0/tokenCount</b>. Here the division by tokenCount normalizes the term frequency. Alternatively if token is absent in the dictionary then a new entry for token is added in dictionary with initial value <b>1.0/tokenCount</b>.

```python
if token in documentFrequency:
	documentFrequency[token] = documentFrequency[token] + 1
else:
	documentFrequency[token] = 1
```
The above block is executed if new word is to be added in term frequency dictionary. While adding new word in term frequency dictionary, similar code is executed to update count in document frequency dictionary.
Finally this method returns both term frequency and document frequency. New term frequency is obtained for new documents however the dictionary of document frequency is updated across documents.

# 7 TF-IDF calculation
```python

def generateIDF(documentFrequency, totalDocuments):
	inverseDocumentFrequency = {}
	for token in documentFrequency:
		inverseDocumentFrequency[token] = math.log((totalDocuments * 1.0)/documentFrequency[token])
	return inverseDocumentFrequency
```
The above block contains generateIDF procedure. This method calculates inverse document frequency by using previously calculated document frequency and total no of documents in corpus.
```python
def generateTfIdf(termFrequency, inverseDocumentFrequency, metaData):
	tfidf = {}

	for token in termFrequency:
		tfidf[metaData["dictionary"][token]] = termFrequency[token] * inverseDocumentFrequency[token]
	return tfidf
```
This procedure calculates the product of term frequency and inverse document frequency.


# 8 Wrapping all for multiprocess execution
All the above methods are for single process execution but they can be easily extended to distributed environment using MPI. There are no changes in core methods themselves but the way they are used in distirbuted environment changes.  
## 8.1 Data distribution
As mentioned before in this project each worker will try to generate Tf-IDF vector for subset of data available to it. For this it is necessary that some type of distributed data storage is available from where the workers read data. Here the workers read data from DFS.
```python
import sys
from mpi4py import MPI

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

	if not os.path.exists(outputPath):
		os.makedirs(outputPath)

	if not os.path.exists(os.path.join(outputPath, "TF")):
		os.makedirs(os.path.join(outputPath, "TF"))

	if not os.path.exists(os.path.join(outputPath, "TFIDF")):
		os.makedirs(os.path.join(outputPath, "TFIDF"))

	pathList = buildInput(inputPath)
	taskChunks = np.array_split(pathList, processorCount)

```
In the initial block, all the required folders for output are created by the root process. Additionally the root process generates list of all files that are to be read and processed. This list is broken in n splits where n is the total no of available workers. This list is only file paths and all workers are responsible for reading the text from DFS. This behavior reduces communication overhead.

```python
"""Every worker will have a copy of stopwords.
These stopwords are loaded from a file.
Source for stopwords: https://github.com/saurabbhsp/stopwords"""
with open('stopwords', 'r') as inputFile:
	stopWords = inputFile.readlines()

stopWords = [word.strip().lower() for word in stopWords]
stopWords.append("")

"""Divide task among workers"""
inputFiles = comm.scatter(taskChunks, root)
```
In above block each worker will read stopwords from available stopwords file. Then the file list that is to be consumed by each workers (taskchunk) is scattered. Now each worker has a copy of inputFile having list of paths from which they are supposed to read data and generate term frequency and inverse document frequency vectors.

## 8.2 Parallel execution
Once the task is divided among the workers, the workers need to start generating term frequency vectors and inverse document frequency vectors in parallel.
```python
"""Calculate term frequency and the relative document frequency
Here first I am reading text from the input file.
Followed by clean up of text file. The cleanText procedure will remove 
characters. This data cleanup script is written for news20 data set and 
hence there are some methods specific to it."""
for file in inputFiles:

	text = readFile(file)
	text = cleanText(text)
	tokens = ngramGenerator(text, (ngramS, ngramE), stopWords)
	tokenTermFrequency, documentFrequency = generateNormalizedTermFrequency(tokens,
		documentFrequency)
	writeToFile(os.path.join(outputPath, "TF",os.path.basename(file)+".json"),
	json.dumps(tokenTermFrequency, indent=2))
	termFrequencies.append(tokenTermFrequency)
```
The above code when executed in parallel by all the workers. They will iteratively read the text from DFS, followed by cleaning the text and generation of tokens based upon the token range. Stopwords are removed withing ngramGeneration method. These methods perform text cleanup and tokenization process. Once tokens are generated normalized term frequency vector and document frequency vector is generated by all workers. The normalized term frequency vectors are dumped on filesystem in json format as well as one copy of term frequency vector is kept in memory which will be later used for TF-IDF calculation.
<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Now we have calculated both document frequency and term frequency on each worker. However the document frequency is not representation of entire corpus but only a sub corpus available with each worker. Hence there is a need to combine all the document frequency from workers.
```python
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
```
In the above block we first gather all the relative document frequencies on root node. Then root combines all the relative document frequencies and then calculates inverse document frequency. This inverse document is sent to all other worker nodes using MPI broadcast. Now each workers have a copy of IDF. Next task is to calculate TF-IDF vector
```python
"""Calculate tfidf"""
for termFrequency, file in zip(termFrequencies, inputFiles):
	tfidf = generateTfIdf(termFrequency, inverseDocumentFrequency)
	writeToFile(os.path.join(outputPath, "TFIDF",os.path.basename(file)+".json"), 
    json.dumps(tfidf, indent=2))
```
# 9 Executing code
To execute the code call the following shell script.
```console
sh run.sh {no of processors} {inputPath} {outputPath} {ngramRangeFrom} {ngramRangeTo}
```
The above shell script expects five parameters. Total no of processors, input and output path on DFS, and ngram range(from, to). This shell script contains following code.
```console
if test "$#" -ne 5; then
    echo "Illegal number of parameters"
else
    mpiexec -n $1 python3 TfIdf.py $2 $3 $4 $5
fi
```
# 10 Performance
<table>
	<th>Processors</th><th>Execution Time (Min)</th>
	<tr><td>2</td><td>2.44</td></tr>
	<tr><td>4</td><td>2.01</td></tr>
	<tr><td>6</td><td>2.58</td></tr>
	<tr><td>8</td><td>2.60</td></tr>
</table>