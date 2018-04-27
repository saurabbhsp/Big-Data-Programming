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
		inverseDocumentFrequency[token] = math.log(
			(totalDocuments * 1.0)/documentFrequency[token])
	return inverseDocumentFrequency
```
The above block contains generateIDF procedure. This method calculates inverse document frequency by using previously calculated document frequency and total no of documents in corpus.
```python
def generateTfIdf(termFrequency, inverseDocumentFrequency):
	tfidf = {}

	for token in termFrequency:
		tfidf[token] = termFrequency[token] * inverseDocumentFrequency[token]
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
	totalDocuments = len(pathList)
	for relativeDocumentFrequency in relativeDocumentFrequencies:
		for token in relativeDocumentFrequency:
			if token in documentFrequency:
				documentFrequency[token] = documentFrequency[token] + 
									relativeDocumentFrequency[token]
			else:
				documentFrequency[token] = relativeDocumentFrequency[token]

	inverseDocumentFrequency = generateIDF(documentFrequency, totalDocuments)

"""Broadcast the idf to every node"""
inverseDocumentFrequency = comm.bcast(inverseDocumentFrequency, root)
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

# 11 Sample output
Following is the output term frequency for item id 8514
```json
{
  "thanks robert": 0.0014814814814814814,
  "work by": 0.0014814814814814814,
  "by navy": 0.0014814814814814814,
  "designate one": 0.0014814814814814814,
  "scientific visualization or": 0.0014814814814814814,
  "authors should submit": 0.0014814814814814814,
  "published": 0.0014814814814814814,
  "numbers and": 0.0014814814814814814,
  "navy organizations will": 0.0014814814814814814,
  "presentations presentations are": 0.0014814814814814814,
  "basin cdnswc or": 0.0014814814814814814,
  "presentation": 0.0044444444444444444,
  "work": 0.002962962962962963,
  "above address": 0.0014814814814814814,
  "warfare center formerly": 0.0014814814814814814,
  "considered four": 0.0014814814814814814,
  "bethesda": 0.005925925925925926,
  "virtual reality seminar": 0.002962962962962963,
  "presentations navy scientific": 0.0014814814814814814,
  "model basin cdnswc": 0.0014814814814814814,
  "voicenet structures": 0.0014814814814814814,
  "attendees abstracts authors": 0.0014814814814814814,
  "minutes in": 0.0014814814814814814,
  "david taylor model": 0.0014814814814814814,
  "engineering": 0.0014814814814814814,
  "april notification": 0.0014814814814814814,
  "organizations will": 0.0014814814814814814,
  "tuesday": 0.0014814814814814814,
  "are solicited": 0.0014814814814814814,
  "include": 0.0014814814814814814,
  "contact deadlines the": 0.0014814814814814814,
  "maryland voice fax": 0.0014814814814814814,
  "scivizvr seminar mar": 0.0014814814814814814,
  "available regular": 0.0014814814814814814,
  "a standalone videotape": 0.0014814814814814814,
  "will be reproduced": 0.0014814814814814814,
  "md call": 0.0014814814814814814,
  "standalone videotape author": 0.0014814814814814814,
  "and fax numbers": 0.0014814814814814814,
  "the abstact": 0.0014814814814814814,
  "virtual reality": 0.007407407407407407,
  "gmtapr gmt": 0.0014814814814814814,
  "engineering software system": 0.0014814814814814814,
  "group code factsnet": 0.0014814814814814814,
  "gmt robert lipman": 0.0014814814814814814,
  "robert lipman at": 0.0014814814814814814,
  "contact robert lipman": 0.0014814814814814814,
  "scivizvr seminar": 0.0014814814814814814,
  "center carderock division": 0.0014814814814814814,
  "applications presentations": 0.0014814814814814814,
  "sheeps sick ": 0.0014814814814814814,
  "possible thanks robert": 0.0014814814814814814,
  "visualization or virtual": 0.0014814814814814814,
  "point of": 0.0014814814814814814,
  "sick": 0.002962962962962963,
  "cdnswc or": 0.0014814814814814814,
  "not be published": 0.0014814814814814814,
  "lipman internet": 0.0014814814814814814,
  "regular presentation minutes": 0.0014814814814814814,
  "should include": 0.0014814814814814814,
  "usa": 0.0014814814814814814,
  "software system": 0.0014814814814814814,
  "the david taylor": 0.0014814814814814814,
  "reality seminar": 0.002962962962962963,
  "include the": 0.0014814814814814814,
  "worksinprogress": 0.0014814814814814814,
  "presentation their affiliations": 0.0014814814814814814,
  "videotape to robert": 0.0014814814814814814,
  "oneday navy scientific": 0.0014814814814814814,
  "author need": 0.0014814814814814814,
  "of acceptance": 0.0014814814814814814,
  "considered four types": 0.0014814814814814814,
  "center carderock": 0.0014814814814814814,
  "presentation minutes": 0.0014814814814814814,
  "naval surface warfare": 0.002962962962962963,
  "be received": 0.0014814814814814814,
  "designate one point": 0.0014814814814814814,
  "the seminar": 0.002962962962962963,
  "seminar is": 0.0014814814814814814,
  "distribute": 0.0014814814814814814,
  "presentations": 0.008888888888888889,
  "usa carderock division": 0.0014814814814814814,
  "standalone": 0.0014814814814814814,
  "page abstract andor": 0.0014814814814814814,
  "presentations navy": 0.002962962962962963,
  "addresses multiauthor": 0.0014814814814814814,
  "ness": 0.0014814814814814814,
  "to robert lipman": 0.0014814814814814814,
  "maryland": 0.0044444444444444444,
  "published in": 0.0014814814814814814,
  "research center": 0.0014814814814814814,
  "of navyrelated": 0.0014814814814814814,
  "of presentations": 0.0014814814814814814,
  "short": 0.0014814814814814814,
  "reproduced for seminar": 0.0014814814814814814,
  "navy engineering software": 0.0014814814814814814,
  "signatures and": 0.0014814814814814814,
  "papers": 0.0014814814814814814,
  "videotape author need": 0.0014814814814814814,
  "in length short": 0.0014814814814814814,
  "demonstration byoh accepted": 0.0014814814814814814,
  "exchange": 0.0014814814814814814,
  "all current": 0.0014814814814814814,
  "of contact deadlines": 0.0014814814814814814,
  "contact robert": 0.0014814814814814814,
  "address please distribute": 0.0014814814814814814,
  "email": 0.0014814814814814814,
  "structures": 0.0014814814814814814,
  "and virtual": 0.005925925925925926,
  "numbers": 0.0014814814814814814,
  "abstact": 0.0014814814814814814,
  "accepted presentations will": 0.0014814814814814814,
  "sick ": 0.002962962962962963,
  "submission deadline is": 0.0014814814814814814,
  "navy organizations": 0.0014814814814814814,
  "surface warfare": 0.002962962962962963,
  "information for navyrelated": 0.0014814814814814814,
  "submit a": 0.0014814814814814814,
  "accepted": 0.0014814814814814814,
  "numbers and addresses": 0.0014814814814814814,
  "june carderock": 0.0014814814814814814,
  "andor videotape to": 0.0014814814814814814,
  "system is sponsoring": 0.0014814814814814814,
  "presentations are available": 0.0014814814814814814,
  "are available regular": 0.0014814814814814814,
  "however viewgraphs": 0.0014814814814814814,
  "aspects": 0.0014814814814814814,
  "for reproduction": 0.0014814814814814814,
  "published in any": 0.0014814814814814814,
  "fax numbers and": 0.0014814814814814814,
  "research developments": 0.0014814814814814814,
  "videotape": 0.002962962962962963,
  "code bethesda": 0.0014814814814814814,
  "multiauthor papers": 0.0014814814814814814,
  "presentations will": 0.0014814814814814814,
  "abstracts authors should": 0.0014814814814814814,
  "solicited on all": 0.0014814814814814814,
  "thanks robert lipman": 0.0014814814814814814,
  "affiliations addresses telephone": 0.0014814814814814814,
  "system is": 0.0014814814814814814,
  "sponsoring": 0.0014814814814814814,
  "sick shieks sixth": 0.0014814814814814814,
  "abstract andor": 0.0014814814814814814,
  "length short presentationminutes": 0.0014814814814814814,
  "the seminar scientific": 0.0014814814814814814,
  "and fax": 0.0014814814814814814,
  "reality all": 0.0014814814814814814,
  "presentations navy scivizvr": 0.0014814814814814814,
  "maryland phishnet": 0.0014814814814814814,
  "affiliations": 0.0014814814814814814,
  "june": 0.002962962962962963,
  "carderock division nswc": 0.0014814814814814814,
  "type": 0.0014814814814814814,
  "developments": 0.0014814814814814814,
  "address please": 0.0014814814814814814,
  "one page abstract": 0.0014814814814814814,
  "computational signatures": 0.0014814814814814814,
  "proceedings however viewgraphs": 0.0014814814814814814,
  "sick shieks": 0.0014814814814814814,
  "fax": 0.002962962962962963,
  "carderock division code": 0.0014814814814814814,
  "short presentationminutes": 0.0014814814814814814,
  "of presentation": 0.0014814814814814814,
  "distribute as widely": 0.0014814814814814814,
  "seminar tuesday": 0.0014814814814814814,
  "phishnet the": 0.0014814814814814814,
  "april notification of": 0.0014814814814814814,
  "mar": 0.0014814814814814814,
  "aspects of navyrelated": 0.0014814814814814814,
  "virtual reality demonstration": 0.0014814814814814814,
  "deadlines the abstact": 0.0014814814814814814,
  "a standalone": 0.0014814814814814814,
  "taylor model basin": 0.0014814814814814814,
  "notification": 0.0014814814814814814,
  "be considered": 0.0014814814814814814,
  "seminar": 0.008888888888888889,
  "and proposed": 0.0014814814814814814,
  "nswc": 0.0014814814814814814,
  "contact": 0.002962962962962963,
  "call for": 0.002962962962962963,
  "division code bethesda": 0.0014814814814814814,
  "robert lipman naval": 0.0014814814814814814,
  "taylor model": 0.0014814814814814814,
  "bethesda maryland": 0.0044444444444444444,
  "visualization and": 0.005925925925925926,
  "their affiliations": 0.0014814814814814814,
  "virtual": 0.007407407407407407,
  "videotape author": 0.0014814814814814814,
  "presentationminutes": 0.0014814814814814814,
  "addresses": 0.002962962962962963,
  "presentations will not": 0.0014814814814814814,
  "on all aspects": 0.0014814814814814814,
  "deadline is april": 0.0014814814814814814,
  "fax numbers": 0.0014814814814814814,
  "proposed work by": 0.0014814814814814814,
  "shieks sixth sheeps": 0.0014814814814814814,
  "developments and applications": 0.0014814814814814814,
  "for seminar attendees": 0.0014814814814814814,
  "maryland sponsor ness": 0.0014814814814814814,
  "center": 0.0044444444444444444,
  "signatures and voicenet": 0.0014814814814814814,
  "nswc bethesda md": 0.0014814814814814814,
  "structures group": 0.0014814814814814814,
  "sheeps sick": 0.0014814814814814814,
  "proposed": 0.0014814814814814814,
  "seminar scientific visualization": 0.0014814814814814814,
  "proceedings": 0.0014814814814814814,
  "engineering software": 0.0014814814814814814,
  "shieks sixth": 0.0014814814814814814,
  "point": 0.0014814814814814814,
  "purpose of": 0.0014814814814814814,
  "division code": 0.0014814814814814814,
  "authors should": 0.002962962962962963,
  "lipman": 0.005925925925925926,
  "sixth sick": 0.0014814814814814814,
  "presentation a standalone": 0.0014814814814814814,
  "scivizvr": 0.0014814814814814814,
  "composmswindowsmisc usa": 0.0014814814814814814,
  "multiauthor": 0.0014814814814814814,
  "presentations presentations": 0.0014814814814814814,
  "worksinprogress and proposed": 0.0014814814814814814,
  "division nswc": 0.0014814814814814814,
  "the abstact submission": 0.0014814814814814814,
  "robert lipman internet": 0.0014814814814814814,
  "center formerly the": 0.0014814814814814814,
  "for navyrelated scientific": 0.0014814814814814814,
  "and virtual reality": 0.005925925925925926,
  "seminar mar gmtapr": 0.0014814814814814814,
  "taylor research center": 0.0014814814814814814,
  "factsnet bethesda maryland": 0.0014814814814814814,
  "abstact submission": 0.0014814814814814814,
  "sponsoring a oneday": 0.0014814814814814814,
  "sixth sheeps sick": 0.0014814814814814814,
  "current": 0.0014814814814814814,
  "for presentations navy": 0.002962962962962963,
  "received by june": 0.0014814814814814814,
  "lipman naval surface": 0.0014814814814814814,
  "reproduction": 0.0014814814814814814,
  "phishnet": 0.0014814814814814814,
  "information contact robert": 0.0014814814814814814,
  "reality demonstration": 0.0014814814814814814,
  "submit a one": 0.0014814814814814814,
  "the type": 0.0014814814814814814,
  "andor videotape": 0.0014814814814814814,
  "video presentation a": 0.0014814814814814814,
  "one point": 0.0014814814814814814,
  "exchange information for": 0.0014814814814814814,
  "cdnswc": 0.0014814814814814814,
  "byoh accepted": 0.0014814814814814814,
  "the david": 0.0014814814814814814,
  "by may materials": 0.0014814814814814814,
  "fax email authors": 0.0014814814814814814,
  "presentation their": 0.0014814814814814814,
  "voice fax": 0.0014814814814814814,
  "attend": 0.0014814814814814814,
  "internet": 0.0014814814814814814,
  "and addresses": 0.0014814814814814814,
  "scientific visualization and": 0.005925925925925926,
  "center bethesda": 0.0014814814814814814,
  "reproduction must be": 0.0014814814814814814,
  "length video presentation": 0.0014814814814814814,
  "signatures": 0.0014814814814814814,
  "deadlines": 0.0014814814814814814,
  "gmt robert": 0.0014814814814814814,
  "the purpose": 0.0014814814814814814,
  "type of": 0.0014814814814814814,
  "and voicenet": 0.0014814814814814814,
  "affiliations addresses": 0.0014814814814814814,
  "must be received": 0.0014814814814814814,
  "seminar the purpose": 0.0014814814814814814,
  "warfare center carderock": 0.0014814814814814814,
  "shieks": 0.0014814814814814814,
  "is april notification": 0.0014814814814814814,
  "and addresses multiauthor": 0.0014814814814814814,
  "for presentations": 0.002962962962962963,
  "telephone and": 0.0014814814814814814,
  "call for presentations": 0.002962962962962963,
  "voicenet": 0.0014814814814814814,
  "videotape to": 0.0014814814814814814,
  "deadline is": 0.0014814814814814814,
  "seminar attendees abstracts": 0.0014814814814814814,
  "robert lipman": 0.005925925925925926,
  "please distribute": 0.0014814814814814814,
  "seminar tuesday june": 0.0014814814814814814,
  "june carderock division": 0.0014814814814814814,
  "received": 0.0014814814814814814,
  "notification of acceptance": 0.0014814814814814814,
  "code factsnet bethesda": 0.0014814814814814814,
  "presentationminutes in": 0.0014814814814814814,
  "is sponsoring": 0.0014814814814814814,
  "email authors should": 0.0014814814814814814,
  "abstract": 0.0014814814814814814,
  "sponsor": 0.0014814814814814814,
  "in any proceedings": 0.0014814814814814814,
  "and other materials": 0.0014814814814814814,
  "authors": 0.002962962962962963,
  "will be considered": 0.0014814814814814814,
  "carderock division": 0.0044444444444444444,
  "by navy organizations": 0.0014814814814814814,
  "division naval surface": 0.0014814814814814814,
  "seminar scientific": 0.0014814814814814814,
  "scientific visualization": 0.007407407407407407,
  "current work": 0.0014814814814814814,
  "md": 0.0014814814814814814,
  "fax email": 0.0014814814814814814,
  "submission deadline": 0.0014814814814814814,
  "regular presentation": 0.0014814814814814814,
  "navy scientific visualization": 0.002962962962962963,
  "organizations": 0.0014814814814814814,
  "bethesda md call": 0.0014814814814814814,
  "or virtual reality": 0.0014814814814814814,
  "all current work": 0.0014814814814814814,
  "md call for": 0.0014814814814814814,
  "nswc bethesda": 0.0014814814814814814,
  "tuesday june carderock": 0.0014814814814814814,
  "exchange information": 0.0014814814814814814,
  "virtual reality all": 0.0014814814814814814,
  "to robert": 0.0014814814814814814,
  "video": 0.0014814814814814814,
  "naval": 0.002962962962962963,
  "viewgraphs and other": 0.0014814814814814814,
  "bethesda maryland voice": 0.0014814814814814814,
  "author need not": 0.0014814814814814814,
  "and exchange": 0.0014814814814814814,
  "email authors": 0.0014814814814814814,
  "navy engineering": 0.0014814814814814814,
  "taylor research": 0.0014814814814814814,
  "taylor": 0.002962962962962963,
  "other materials": 0.0014814814814814814,
  "cdnswc or computational": 0.0014814814814814814,
  "should designate": 0.0014814814814814814,
  "author": 0.0014814814814814814,
  "maryland phishnet the": 0.0014814814814814814,
  "attend the seminar": 0.0014814814814814814,
  "length short": 0.0014814814814814814,
  "june for further": 0.0014814814814814814,
  "oneday navy": 0.0014814814814814814,
  "phishnet the sixth": 0.0014814814814814814,
  "abstract andor videotape": 0.0014814814814814814,
  "seminar the": 0.0014814814814814814,
  "voicenet structures group": 0.0014814814814814814,
  "code": 0.002962962962962963,
  "solicited on": 0.0014814814814814814,
  "structures group code": 0.0014814814814814814,
  "gmtapr gmt robert": 0.0014814814814814814,
  "demonstration byoh": 0.0014814814814814814,
  "acceptance will be": 0.0014814814814814814,
  "sixth sick shieks": 0.0014814814814814814,
  "navyrelated scientific": 0.002962962962962963,
  "authors should include": 0.0014814814814814814,
  "surface": 0.002962962962962963,
  "be published": 0.0014814814814814814,
  "not attend": 0.0014814814814814814,
  "bethesda md": 0.0014814814814814814,
  "minutes": 0.0014814814814814814,
  "and voicenet structures": 0.0014814814814814814,
  "ness navy": 0.0014814814814814814,
  "types of presentations": 0.0014814814814814814,
  "seminar is to": 0.0014814814814814814,
  "aspects of": 0.0014814814814814814,
  "center formerly": 0.0014814814814814814,
  "addresses telephone and": 0.0014814814814814814,
  "reality programs": 0.0014814814814814814,
  "byoh accepted presentations": 0.0014814814814814814,
  "voice": 0.0014814814814814814,
  "papers should designate": 0.0014814814814814814,
  "for navyrelated": 0.0014814814814814814,
  "voice fax email": 0.0014814814814814814,
  "their affiliations addresses": 0.0014814814814814814,
  "work by navy": 0.0014814814814814814,
  "lipman at": 0.0014814814814814814,
  "viewgraphs and": 0.0014814814814814814,
  "sponsoring a": 0.0014814814814814814,
  "of contact": 0.0014814814814814814,
  "mar gmtapr": 0.0014814814814814814,
  "ness navy engineering": 0.0014814814814814814,
  "abstracts authors": 0.0014814814814814814,
  "division nswc bethesda": 0.0014814814814814814,
  "all aspects": 0.0014814814814814814,
  "code bethesda maryland": 0.0014814814814814814,
  "video presentation": 0.0014814814814814814,
  "system": 0.0014814814814814814,
  "or virtual": 0.0014814814814814814,
  "composmswindowsmisc": 0.0014814814814814814,
  "abstracts": 0.0014814814814814814,
  "factsnet": 0.0014814814814814814,
  "materials will be": 0.0014814814814814814,
  "for seminar": 0.0014814814814814814,
  "demonstration": 0.0014814814814814814,
  "further information contact": 0.0014814814814814814,
  "proposed work": 0.0014814814814814814,
  "materials": 0.002962962962962963,
  "types of": 0.0014814814814814814,
  "internet david taylor": 0.0014814814814814814,
  "notification of": 0.0014814814814814814,
  "basin cdnswc": 0.0014814814814814814,
  "organizations will be": 0.0014814814814814814,
  "sponsor ness navy": 0.0014814814814814814,
  "regular": 0.0014814814814814814,
  "reality programs research": 0.0014814814814814814,
  "group code": 0.0014814814814814814,
  "or computational": 0.0014814814814814814,
  "developments and": 0.0014814814814814814,
  "need not attend": 0.0014814814814814814,
  "include the type": 0.0014814814814814814,
  "papers should": 0.0014814814814814814,
  "reality seminar tuesday": 0.0014814814814814814,
  "and applications presentations": 0.0014814814814814814,
  "submission": 0.0014814814814814814,
  "sponsor ness": 0.0014814814814814814,
  "division": 0.0044444444444444444,
  "accepted presentations": 0.0014814814814814814,
  "reality all current": 0.0014814814814814814,
  "in length video": 0.0014814814814814814,
  "should submit": 0.0014814814814814814,
  "or computational signatures": 0.0014814814814814814,
  "navy": 0.007407407407407407,
  "types": 0.0014814814814814814,
  "deadline": 0.0014814814814814814,
  "warfare": 0.002962962962962963,
  "usa carderock": 0.0014814814814814814,
  "maryland voice": 0.0014814814814814814,
  "sheeps": 0.0014814814814814814,
  "telephone and fax": 0.0014814814814814814,
  "by june": 0.0014814814814814814,
  "programs research": 0.0014814814814814814,
  "short presentationminutes in": 0.0014814814814814814,
  "computational signatures and": 0.0014814814814814814,
  "submit": 0.0014814814814814814,
  "programs research developments": 0.0014814814814814814,
  "basin": 0.0014814814814814814,
  "multiauthor papers should": 0.0014814814814814814,
  "division naval": 0.0014814814814814814,
  "call": 0.002962962962962963,
  "presentation minutes in": 0.0014814814814814814,
  "composmswindowsmisc usa carderock": 0.0014814814814814814,
  "code factsnet": 0.0014814814814814814,
  "purpose of the": 0.0014814814814814814,
  "david taylor": 0.002962962962962963,
  "gmt": 0.0014814814814814814,
  "materials for reproduction": 0.0014814814814814814,
  "center bethesda maryland": 0.0014814814814814814,
  "navyrelated scientific visualization": 0.002962962962962963,
  "in length": 0.002962962962962963,
  "and proposed work": 0.0014814814814814814,
  "a oneday navy": 0.0014814814814814814,
  "visualization and virtual": 0.005925925925925926,
  "maryland sponsor": 0.0014814814814814814,
  "four types": 0.0014814814814814814,
  "david": 0.002962962962962963,
  "visualization": 0.007407407407407407,
  "attendees": 0.0014814814814814814,
  "byoh": 0.0014814814814814814,
  "information contact": 0.0014814814814814814,
  "work worksinprogress and": 0.0014814814814814814,
  "presentations are solicited": 0.0014814814814814814,
  "sixth sheeps": 0.0014814814814814814,
  "software": 0.0014814814814814814,
  "virtual reality programs": 0.0014814814814814814,
  "the above address": 0.0014814814814814814,
  "reproduced for": 0.0014814814814814814,
  "mar gmtapr gmt": 0.0014814814814814814,
  "present and exchange": 0.0014814814814814814,
  "lipman naval": 0.0014814814814814814,
  "purpose": 0.0014814814814814814,
  "viewgraphs": 0.0014814814814814814,
  "minutes in length": 0.0014814814814814814,
  "point of contact": 0.0014814814814814814,
  "abstact submission deadline": 0.0014814814814814814,
  "length video": 0.0014814814814814814,
  "reproduced": 0.0014814814814814814,
  "internet david": 0.0014814814814814814,
  "addresses multiauthor papers": 0.0014814814814814814,
  "solicited": 0.0014814814814814814,
  "received by": 0.0014814814814814814,
  "warfare center": 0.002962962962962963,
  "lipman at the": 0.0014814814814814814,
  "navy scientific": 0.002962962962962963,
  "proceedings however": 0.0014814814814814814,
  "model basin": 0.0014814814814814814,
  "be reproduced": 0.0014814814814814814,
  "deadlines the": 0.0014814814814814814,
  "carderock": 0.0044444444444444444,
  "reality": 0.007407407407407407,
  "designate": 0.0014814814814814814,
  "visualization or": 0.0014814814814814814,
  "applications presentations presentations": 0.0014814814814814814,
  "navy scivizvr": 0.0014814814814814814,
  "programs": 0.0014814814814814814,
  "a oneday": 0.0014814814814814814,
  "may materials": 0.0014814814814814814,
  "tuesday june": 0.0014814814814814814,
  "page abstract": 0.0014814814814814814,
  "presentation a": 0.0014814814814814814,
  "software system is": 0.0014814814814814814,
  "worksinprogress and": 0.0014814814814814814,
  "work worksinprogress": 0.0014814814814814814,
  "considered": 0.0014814814814814814,
  "of navyrelated scientific": 0.0014814814814814814,
  "gmtapr": 0.0014814814814814814,
  "navyrelated": 0.002962962962962963,
  "seminar mar": 0.0014814814814814814,
  "factsnet bethesda": 0.0014814814814814814,
  "model": 0.0014814814814814814,
  "and applications": 0.0014814814814814814,
  "the sixth": 0.0014814814814814814,
  "reality seminar the": 0.0014814814814814814,
  "materials for": 0.0014814814814814814,
  "seminar attendees": 0.0014814814814814814,
  "distribute as": 0.0014814814814814814,
  "any proceedings": 0.0014814814814814814,
  "robert lipman composmswindowsmisc": 0.0014814814814814814,
  "attendees abstracts": 0.0014814814814814814,
  "presentationminutes in length": 0.0014814814814814814,
  "addresses telephone": 0.0014814814814814814,
  "materials will": 0.0014814814814814814,
  "length": 0.002962962962962963,
  "sixth": 0.002962962962962963,
  "surface warfare center": 0.002962962962962963,
  "reality demonstration byoh": 0.0014814814814814814,
  "standalone videotape": 0.0014814814814814814,
  "computational": 0.0014814814814814814,
  "contact deadlines": 0.0014814814814814814,
  "is april": 0.0014814814814814814,
  "lipman composmswindowsmisc usa": 0.0014814814814814814,
  "available regular presentation": 0.0014814814814814814,
  "lipman internet david": 0.0014814814814814814,
  "group": 0.0014814814814814814,
  "the sixth sick": 0.0014814814814814814,
  "type of presentation": 0.0014814814814814814,
  "oneday": 0.0014814814814814814,
  "bethesda maryland phishnet": 0.0014814814814814814,
  "presentations are": 0.002962962962962963,
  "david taylor research": 0.0014814814814814814,
  "research center bethesda": 0.0014814814814814814,
  "reproduction must": 0.0014814814814814814,
  "acceptance will": 0.0014814814814814814,
  "of the seminar": 0.0014814814814814814,
  "lipman composmswindowsmisc": 0.0014814814814814814,
  "robert": 0.005925925925925926,
  "attend the": 0.0014814814814814814,
  "navy scivizvr seminar": 0.0014814814814814814,
  "naval surface": 0.002962962962962963,
  "scientific": 0.007407407407407407,
  "address": 0.0014814814814814814,
  "acceptance": 0.0014814814814814814,
  "telephone": 0.0014814814814814814,
  "carderock division naval": 0.0014814814814814814,
  "andor": 0.0014814814814814814,
  "bethesda maryland sponsor": 0.0014814814814814814,
  "applications": 0.0014814814814814814,
  "current work worksinprogress": 0.0014814814814814814,
  "june for": 0.0014814814814814814,
  "april": 0.0014814814814814814,
  "formerly the david": 0.0014814814814814814
}
```
Following is the output inverse document frequency for item id 8514
```json
{
  "thanks robert": 0.011416463650294494,
  "work by": 0.010474258069968573,
  "by navy": 0.013044037411284288,
  "designate one": 0.013044037411284288,
  "scientific visualization or": 0.013044037411284288,
  "published": 0.00611671999368977,
  "will be reproduced": 0.013044037411284288,
  "is sponsoring": 0.011590957036452099,
  "presentations presentations are": 0.013044037411284288,
  "basin cdnswc or": 0.013044037411284288,
  "presentation": 0.02519658238527886,
  "presentation their affiliations": 0.013044037411284288,
  "above address": 0.010659684948418952,
  "warfare center formerly": 0.013044037411284288,
  "notification of": 0.012287258709408742,
  "considered four": 0.013044037411284288,
  "materials will be": 0.013044037411284288,
  "virtual reality seminar": 0.026088074822568575,
  "presentations navy scientific": 0.013044037411284288,
  "model basin cdnswc": 0.013044037411284288,
  "voicenet structures": 0.013044037411284288,
  "minutes in": 0.011260373997468084,
  "andor": 0.006018808716906081,
  "engineering": 0.004593989300682506,
  "carderock division code": 0.013044037411284288,
  "organizations will": 0.012617841748392759,
  "tuesday": 0.007530453297303653,
  "are solicited": 0.012017152699343626,
  "for navyrelated": 0.013044037411284288,
  "include": 0.0048899994801417275,
  "contact deadlines the": 0.013044037411284288,
  "maryland voice fax": 0.013044037411284288,
  "scivizvr seminar mar": 0.013644726460333418,
  "available regular": 0.013044037411284288,
  "a standalone videotape": 0.013044037411284288,
  "numbers and": 0.00863081495908042,
  "md call": 0.013044037411284288,
  "standalone videotape author": 0.013044037411284288,
  "the abstact": 0.013044037411284288,
  "virtual reality": 0.045850114626523664,
  "solicited": 0.010761896609881102,
  "gmtapr gmt": 0.012287258709408742,
  "visualization": 0.04723686679013956,
  "engineering software system": 0.013044037411284288,
  "abstact submission deadline": 0.013044037411284288,
  "gmt robert lipman": 0.013044037411284288,
  "robert lipman at": 0.013044037411284288,
  "division naval surface": 0.012287258709408742,
  "center carderock division": 0.013044037411284288,
  "applications presentations": 0.013044037411284288,
  "sheeps sick ": 0.013044037411284288,
  "attend the": 0.010474258069968573,
  "reproduction must": 0.013044037411284288,
  "reproduced": 0.010309479351286758,
  "sick": 0.013282377902259902,
  "cdnswc or": 0.013044037411284288,
  "not be published": 0.012287258709408742,
  "internet david": 0.012017152699343626,
  "lipman internet": 0.013044037411284288,
  "regular presentation minutes": 0.013044037411284288,
  "should include": 0.010990267987402966,
  "usa": 0.0034371703955283573,
  "software system": 0.012017152699343626,
  "include the": 0.007542449164314941,
  "worksinprogress": 0.013044037411284288,
  "visualization or": 0.013044037411284288,
  "videotape to robert": 0.013044037411284288,
  "author need": 0.013044037411284288,
  "of acceptance": 0.011260373997468084,
  "considered four types": 0.013044037411284288,
  "center carderock": 0.013044037411284288,
  "presentation minutes": 0.013044037411284288,
  "naval surface warfare": 0.021523793219762204,
  "be received": 0.010233489285527425,
  "warfare center": 0.021980535974805933,
  "the seminar": 0.025235683496785517,
  "seminar is": 0.013044037411284288,
  "distribute": 0.008420488646087253,
  "presentations": 0.05906880891789796,
  "usa carderock division": 0.011260373997468084,
  "standalone": 0.00990290624654341,
  "page abstract andor": 0.013044037411284288,
  "presentations navy": 0.026088074822568575,
  "addresses multiauthor": 0.013044037411284288,
  "ness": 0.011788781321821762,
  "to robert lipman": 0.013044037411284288,
  "maryland": 0.020861561726415918,
  "published in": 0.007591428219516257,
  "research center": 0.007591428219516257,
  "of navyrelated": 0.013044037411284288,
  "of presentations": 0.012617841748392759,
  "short": 0.005103702604355871,
  "reproduced for seminar": 0.013044037411284288,
  "navy engineering software": 0.013044037411284288,
  "signatures and": 0.011260373997468084,
  "papers": 0.00682225211072143,
  "proceedings however viewgraphs": 0.013044037411284288,
  "sheeps sick": 0.013044037411284288,
  "deadlines the": 0.012617841748392759,
  "demonstration byoh accepted": 0.013044037411284288,
  "exchange": 0.007152835149705445,
  "organizations will be": 0.012617841748392759,
  "all current": 0.011590957036452099,
  "of contact deadlines": 0.013044037411284288,
  "contact robert": 0.011788781321821762,
  "address please distribute": 0.013044037411284288,
  "structures": 0.00851030290063012,
  "reality": 0.02945490011344142,
  "email": 0.0030947228258970606,
  "and virtual": 0.04914903483763497,
  "numbers": 0.005283721200205277,
  "abstact": 0.012617841748392759,
  "accepted presentations will": 0.013044037411284288,
  "sick ": 0.026088074822568575,
  "submission deadline is": 0.013044037411284288,
  "navy organizations": 0.013044037411284288,
  "surface warfare": 0.021319369896837904,
  "information for navyrelated": 0.013044037411284288,
  "regular": 0.006058084458459166,
  "purpose of": 0.006953853908805306,
  "numbers and addresses": 0.013044037411284288,
  "june carderock": 0.013044037411284288,
  "andor videotape to": 0.013044037411284288,
  "navy scientific visualization": 0.026088074822568575,
  "however viewgraphs": 0.013044037411284288,
  "aspects": 0.0071436047056304295,
  "reality seminar": 0.026088074822568575,
  "research developments": 0.013044037411284288,
  "videotape": 0.019366049514958603,
  "developments and applications": 0.013044037411284288,
  "code bethesda": 0.012617841748392759,
  "by june": 0.012017152699343626,
  "presentation minutes in": 0.013044037411284288,
  "presentations will": 0.012617841748392759,
  "abstracts authors should": 0.013044037411284288,
  "solicited on all": 0.012617841748392759,
  "thanks robert lipman": 0.013044037411284288,
  "system is": 0.0068982421764807655,
  "reality seminar the": 0.013044037411284288,
  "sponsoring": 0.011119173731091307,
  "abstract andor": 0.013044037411284288,
  "length short presentationminutes": 0.013044037411284288,
  "video presentation": 0.013044037411284288,
  "the seminar scientific": 0.013044037411284288,
  "and fax": 0.00990290624654341,
  "reality all": 0.013044037411284288,
  "presentations navy scivizvr": 0.013044037411284288,
  "maryland phishnet": 0.013044037411284288,
  "affiliations": 0.011590957036452099,
  "june": 0.013891560840521532,
  "carderock division nswc": 0.009735011897940442,
  "type": 0.004872017690490729,
  "developments": 0.009447373358027913,
  "address please": 0.010990267987402966,
  "one page abstract": 0.013044037411284288,
  "computational signatures": 0.013044037411284288,
  "sick shieks": 0.013044037411284288,
  "fax": 0.009009124824395856,
  "april notification": 0.013044037411284288,
  "short presentationminutes": 0.013044037411284288,
  "nswc bethesda": 0.009788889889304701,
  "distribute as widely": 0.013044037411284288,
  "seminar tuesday": 0.013044037411284288,
  "phishnet the": 0.013044037411284288,
  "mar": 0.00884668430897878,
  "aspects of navyrelated": 0.013044037411284288,
  "virtual reality demonstration": 0.013044037411284288,
  "formerly the david": 0.013044037411284288,
  "deadlines the abstact": 0.013044037411284288,
  "a standalone": 0.010871686198256986,
  "notification": 0.010871686198256986,
  "affiliations addresses telephone": 0.013044037411284288,
  "exchange information for": 0.013044037411284288,
  "seminar": 0.06395810969051371,
  "and proposed": 0.011788781321821762,
  "nswc": 0.009404428858956428,
  "contact": 0.010311379489634023,
  "call for": 0.01611090199297902,
  "robert lipman naval": 0.013044037411284288,
  "taylor model": 0.011260373997468084,
  "bethesda maryland": 0.03785352524517828,
  "visualization and": 0.04715512528728705,
  "lipman naval": 0.013044037411284288,
  "their affiliations": 0.013044037411284288,
  "virtual": 0.036859880415761465,
  "videotape author": 0.013044037411284288,
  "presentationminutes": 0.013044037411284288,
  "addresses": 0.01438068602677175,
  "presentations will not": 0.013044037411284288,
  "on all aspects": 0.013044037411284288,
  "deadline is april": 0.013044037411284288,
  "fax numbers": 0.013044037411284288,
  "shieks sixth sheeps": 0.013044037411284288,
  "and fax numbers": 0.013044037411284288,
  "for seminar attendees": 0.013044037411284288,
  "maryland sponsor ness": 0.013044037411284288,
  "center": 0.013012993839240226,
  "the david taylor": 0.013044037411284288,
  "structures group": 0.013044037411284288,
  "oneday navy scientific": 0.013044037411284288,
  "proposed": 0.006570781333555365,
  "code": 0.009685049432365382,
  "seminar scientific visualization": 0.013044037411284288,
  "proceedings": 0.00884668430897878,
  "engineering software": 0.012617841748392759,
  "shieks sixth": 0.013044037411284288,
  "worksinprogress and": 0.013044037411284288,
  "accepted": 0.00620194050352412,
  "division code": 0.013044037411284288,
  "authors should": 0.024574517418817485,
  "lipman": 0.05217614964513715,
  "sixth sick": 0.013044037411284288,
  "presentation a standalone": 0.013044037411284288,
  "scivizvr": 0.013044037411284288,
  "composmswindowsmisc usa": 0.013644726460333418,
  "division code bethesda": 0.013044037411284288,
  "presentations presentations": 0.013044037411284288,
  "proposed work by": 0.013044037411284288,
  "division nswc": 0.009735011897940442,
  "of navyrelated scientific": 0.013044037411284288,
  "and virtual reality": 0.04914903483763497,
  "seminar mar gmtapr": 0.014671611172274078,
  "taylor research center": 0.013044037411284288,
  "should submit": 0.013044037411284288,
  "abstact submission": 0.013044037411284288,
  "sponsoring a oneday": 0.013044037411284288,
  "sixth sheeps sick": 0.013044037411284288,
  "current": 0.004724691868704491,
  "robert lipman": 0.05217614964513715,
  "basin cdnswc": 0.013044037411284288,
  "lipman naval surface": 0.013044037411284288,
  "reproduction": 0.009491599970101515,
  "phishnet": 0.013044037411284288,
  "reality demonstration": 0.013044037411284288,
  "a oneday": 0.012287258709408742,
  "submit a one": 0.013044037411284288,
  "the type": 0.007909613851580987,
  "andor videotape": 0.013044037411284288,
  "video presentation a": 0.013044037411284288,
  "one point": 0.008442436921324499,
  "navy engineering": 0.013044037411284288,
  "cdnswc": 0.012617841748392759,
  "byoh accepted": 0.013044037411284288,
  "the david": 0.011119173731091307,
  "by may materials": 0.013044037411284288,
  "fax email authors": 0.013044037411284288,
  "presentation their": 0.013044037411284288,
  "voice fax": 0.008255709927405439,
  "attend": 0.00863081495908042,
  "internet": 0.004132416935879952,
  "and addresses": 0.010871686198256986,
  "scientific visualization and": 0.05217614964513715,
  "center bethesda": 0.012287258709408742,
  "reproduction must be": 0.013044037411284288,
  "length video presentation": 0.013044037411284288,
  "signatures": 0.009735011897940442,
  "attendees abstracts": 0.013044037411284288,
  "deadlines": 0.012287258709408742,
  "gmt robert": 0.009963383275462307,
  "the purpose": 0.007238604006622339,
  "type of": 0.006112126248981254,
  "and voicenet": 0.013044037411284288,
  "affiliations addresses": 0.013044037411284288,
  "must be received": 0.011788781321821762,
  "seminar the purpose": 0.013044037411284288,
  "warfare center carderock": 0.013044037411284288,
  "for reproduction": 0.012617841748392759,
  "is april notification": 0.013044037411284288,
  "the sixth": 0.010474258069968573,
  "for presentations": 0.025235683496785517,
  "telephone and": 0.011260373997468084,
  "call for presentations": 0.026088074822568575,
  "voicenet": 0.012617841748392759,
  "maryland phishnet the": 0.013044037411284288,
  "deadline is": 0.013044037411284288,
  "seminar attendees abstracts": 0.013044037411284288,
  "for presentations navy": 0.026088074822568575,
  "please distribute": 0.013044037411284288,
  "seminar tuesday june": 0.013044037411284288,
  "information contact robert": 0.013044037411284288,
  "received": 0.005780626167887189,
  "notification of acceptance": 0.013044037411284288,
  "code factsnet bethesda": 0.013044037411284288,
  "presentationminutes in": 0.013044037411284288,
  "navy organizations will": 0.013044037411284288,
  "acceptance will": 0.012287258709408742,
  "email authors should": 0.013044037411284288,
  "abstract": 0.008179719861646105,
  "sponsor": 0.010233489285527425,
  "in any proceedings": 0.013044037411284288,
  "and other materials": 0.012287258709408742,
  "authors": 0.01468023007676099,
  "minutes in length": 0.013044037411284288,
  "materials for": 0.011416463650294494,
  "carderock division": 0.029049074272437903,
  "scivizvr seminar": 0.013044037411284288,
  "seminar scientific": 0.013044037411284288,
  "scientific visualization": 0.053809483049405506,
  "contact robert lipman": 0.013044037411284288,
  "md": 0.006168214265341879,
  "fax email": 0.008681905590296966,
  "submission deadline": 0.010990267987402966,
  "regular presentation": 0.013044037411284288,
  "are available regular": 0.013044037411284288,
  "organizations": 0.006829678098608164,
  "bethesda md call": 0.013044037411284288,
  "or virtual reality": 0.013044037411284288,
  "all current work": 0.013044037411284288,
  "of presentation": 0.013044037411284288,
  "tuesday june carderock": 0.013044037411284288,
  "exchange information": 0.012017152699343626,
  "virtual reality all": 0.013044037411284288,
  "robert": 0.020868275697060176,
  "to robert": 0.011416463650294494,
  "video": 0.005177494444778525,
  "naval": 0.016840977292174507,
  "viewgraphs and other": 0.013044037411284288,
  "robert lipman internet": 0.013044037411284288,
  "and exchange": 0.012017152699343626,
  "email authors": 0.013044037411284288,
  "april": 0.005375667437100835,
  "presentations are available": 0.013044037411284288,
  "taylor research": 0.013044037411284288,
  "and applications presentations": 0.013044037411284288,
  "taylor": 0.01635943972329221,
  "other materials": 0.012287258709408742,
  "cdnswc or computational": 0.013044037411284288,
  "standalone videotape": 0.013044037411284288,
  "author": 0.006049266090270579,
  "work worksinprogress": 0.013044037411284288,
  "length short": 0.013044037411284288,
  "june for further": 0.013044037411284288,
  "oneday navy": 0.013044037411284288,
  "presentationminutes in length": 0.013044037411284288,
  "abstract andor videotape": 0.013044037411284288,
  "seminar the": 0.013044037411284288,
  "available regular presentation": 0.013044037411284288,
  "voicenet structures group": 0.013044037411284288,
  "phishnet the sixth": 0.013044037411284288,
  "md call for": 0.013044037411284288,
  "structures group code": 0.013044037411284288,
  "gmtapr gmt robert": 0.014671611172274078,
  "visualization and virtual": 0.05217614964513715,
  "naval surface": 0.021523793219762204,
  "demonstration byoh": 0.013044037411284288,
  "acceptance will be": 0.012617841748392759,
  "sixth sick shieks": 0.013044037411284288,
  "authors should submit": 0.013044037411284288,
  "considered": 0.005073819109742864,
  "fax numbers and": 0.013044037411284288,
  "david taylor model": 0.011260373997468084,
  "not attend": 0.013044037411284288,
  "bethesda md": 0.009404428858956428,
  "minutes": 0.006771287034103161,
  "ness navy": 0.013044037411284288,
  "types of presentations": 0.013044037411284288,
  "division": 0.015773188891056233,
  "seminar is to": 0.013044037411284288,
  "aspects of": 0.007471889091738268,
  "center formerly": 0.013044037411284288,
  "addresses telephone and": 0.013044037411284288,
  "reality programs": 0.013044037411284288,
  "byoh accepted presentations": 0.013044037411284288,
  "papers should designate": 0.013044037411284288,
  "by navy organizations": 0.013044037411284288,
  "voice fax email": 0.011260373997468084,
  "work by navy": 0.013044037411284288,
  "lipman at": 0.013044037411284288,
  "bethesda": 0.03682641829434706,
  "sponsoring a": 0.012287258709408742,
  "applications presentations presentations": 0.013044037411284288,
  "of contact": 0.011119173731091307,
  "attend the seminar": 0.013044037411284288,
  "lipman internet david": 0.013044037411284288,
  "abstracts authors": 0.013044037411284288,
  "division nswc bethesda": 0.009788889889304701,
  "should designate": 0.013044037411284288,
  "code bethesda maryland": 0.013044037411284288,
  "center formerly the": 0.013044037411284288,
  "taylor model basin": 0.011260373997468084,
  "or virtual": 0.012287258709408742,
  "composmswindowsmisc": 0.010233489285527425,
  "abstracts": 0.010309479351286758,
  "navyrelated scientific": 0.026088074822568575,
  "factsnet": 0.013044037411284288,
  "for seminar": 0.013044037411284288,
  "demonstration": 0.008557338009244313,
  "further information contact": 0.011788781321821762,
  "proposed work": 0.013044037411284288,
  "materials": 0.015037107567356153,
  "types of": 0.007199870258319639,
  "is april": 0.011119173731091307,
  "internet david taylor": 0.013044037411284288,
  "surface": 0.013615021759655697,
  "videotape author need": 0.013044037411284288,
  "be published": 0.010026434556082745,
  "sponsor ness navy": 0.013044037411284288,
  "submit a": 0.011119173731091307,
  "reality programs research": 0.013044037411284288,
  "group code": 0.013044037411284288,
  "or computational": 0.013044037411284288,
  "developments and": 0.012287258709408742,
  "need not attend": 0.013044037411284288,
  "tuesday june": 0.012617841748392759,
  "navy scivizvr": 0.013044037411284288,
  "papers should": 0.012287258709408742,
  "reality seminar tuesday": 0.013044037411284288,
  "and voicenet structures": 0.013044037411284288,
  "submission": 0.008656140045538642,
  "sponsor ness": 0.013044037411284288,
  "system": 0.0026049898136253744,
  "accepted presentations": 0.013044037411284288,
  "worksinprogress and proposed": 0.013044037411284288,
  "in length video": 0.013044037411284288,
  "factsnet bethesda maryland": 0.013044037411284288,
  "or computational signatures": 0.013044037411284288,
  "navy": 0.043674106341129865,
  "types": 0.006135238753340483,
  "reality all current": 0.013044037411284288,
  "deadline": 0.009065404307209987,
  "warfare": 0.017635833548751336,
  "usa carderock": 0.011260373997468084,
  "june carderock division": 0.013044037411284288,
  "sheeps": 0.013044037411284288,
  "telephone and fax": 0.012287258709408742,
  "current work": 0.012617841748392759,
  "and proposed work": 0.013044037411284288,
  "author need not": 0.013044037411284288,
  "programs research developments": 0.013044037411284288,
  "basin": 0.010990267987402966,
  "multiauthor papers should": 0.013044037411284288,
  "division naval": 0.012287258709408742,
  "call": 0.007770786210074848,
  "composmswindowsmisc usa carderock": 0.014671611172274078,
  "code factsnet": 0.013044037411284288,
  "purpose of the": 0.00909946285272806,
  "david taylor": 0.022238347462182614,
  "gmt": 0.00024359979861729238,
  "materials for reproduction": 0.013044037411284288,
  "center bethesda maryland": 0.013044037411284288,
  "navyrelated scientific visualization": 0.026088074822568575,
  "in length": 0.02112814464902288,
  "short presentationminutes in": 0.013044037411284288,
  "a oneday navy": 0.013044037411284288,
  "computational signatures and": 0.013044037411284288,
  "maryland sponsor": 0.013044037411284288,
  "attendees abstracts authors": 0.013044037411284288,
  "david": 0.007730252568877574,
  "four types": 0.012617841748392759,
  "attendees": 0.010389578938353834,
  "byoh": 0.013044037411284288,
  "information contact": 0.009584222721184973,
  "work worksinprogress and": 0.013044037411284288,
  "presentations are solicited": 0.013044037411284288,
  "sixth sheeps": 0.013044037411284288,
  "software": 0.00399549666638926,
  "designate": 0.011788781321821762,
  "virtual reality programs": 0.013044037411284288,
  "received by": 0.010092289019150647,
  "reproduced for": 0.013044037411284288,
  "david taylor research": 0.013044037411284288,
  "the above address": 0.010871686198256986,
  "present and exchange": 0.013044037411284288,
  "be considered": 0.006890465352945738,
  "purpose": 0.005562769252396083,
  "address": 0.004770335697064782,
  "viewgraphs": 0.012017152699343626,
  "point of contact": 0.012617841748392759,
  "group code factsnet": 0.013044037411284288,
  "visualization or virtual": 0.013044037411284288,
  "point of": 0.005984750171388008,
  "programs research": 0.013044037411284288,
  "addresses multiauthor papers": 0.013044037411284288,
  "work": 0.006027609377143643,
  "sick shieks sixth": 0.013044037411284288,
  "designate one point": 0.013044037411284288,
  "lipman at the": 0.013044037411284288,
  "navy scientific": 0.026088074822568575,
  "proceedings however": 0.013044037411284288,
  "model basin": 0.011260373997468084,
  "be reproduced": 0.012017152699343626,
  "in length short": 0.013044037411284288,
  "carderock": 0.028898400709434877,
  "shieks": 0.012617841748392759,
  "authors should include": 0.013044037411284288,
  "multiauthor papers": 0.013044037411284288,
  "maryland voice": 0.013044037411284288,
  "include the type": 0.013044037411284288,
  "programs": 0.005158096701449608,
  "received by june": 0.013044037411284288,
  "may materials": 0.013044037411284288,
  "nswc bethesda md": 0.009788889889304701,
  "page abstract": 0.013044037411284288,
  "presentation a": 0.013044037411284288,
  "software system is": 0.013044037411284288,
  "point": 0.0032260238514030385,
  "april notification of": 0.013044037411284288,
  "multiauthor": 0.013044037411284288,
  "the abstact submission": 0.013044037411284288,
  "gmtapr": 0.011416463650294494,
  "navyrelated": 0.026088074822568575,
  "seminar mar": 0.013644726460333418,
  "factsnet bethesda": 0.013044037411284288,
  "model": 0.005528351027961611,
  "and applications": 0.010233489285527425,
  "and addresses multiauthor": 0.013044037411284288,
  "viewgraphs and": 0.012617841748392759,
  "videotape to": 0.013044037411284288,
  "seminar attendees": 0.013044037411284288,
  "distribute as": 0.013044037411284288,
  "any proceedings": 0.013044037411284288,
  "robert lipman composmswindowsmisc": 0.014671611172274078,
  "will be considered": 0.009491599970101515,
  "their affiliations addresses": 0.013044037411284288,
  "addresses telephone": 0.013044037411284288,
  "materials will": 0.013044037411284288,
  "length": 0.013704364094976253,
  "sixth": 0.01769336861795756,
  "surface warfare center": 0.021980535974805933,
  "reality demonstration byoh": 0.013044037411284288,
  "voice": 0.005307509995606549,
  "computational": 0.00953718761257078,
  "contact deadlines": 0.013044037411284288,
  "signatures and voicenet": 0.013044037411284288,
  "lipman composmswindowsmisc usa": 0.014671611172274078,
  "mar gmtapr": 0.014671611172274078,
  "group": 0.0038069054009293615,
  "the sixth sick": 0.013044037411284288,
  "type of presentation": 0.013044037411284288,
  "oneday": 0.011260373997468084,
  "bethesda maryland phishnet": 0.013044037411284288,
  "presentations are": 0.026088074822568575,
  "system is sponsoring": 0.013044037411284288,
  "research center bethesda": 0.013044037411284288,
  "possible thanks robert": 0.013044037411284288,
  "bethesda maryland voice": 0.013044037411284288,
  "of the seminar": 0.013044037411284288,
  "lipman composmswindowsmisc": 0.014671611172274078,
  "all aspects": 0.00953718761257078,
  "length video": 0.013044037411284288,
  "submit": 0.00780534600452647,
  "navy scivizvr seminar": 0.013044037411284288,
  "solicited on": 0.012617841748392759,
  "scientific": 0.02984005044765165,
  "ness navy engineering": 0.013044037411284288,
  "acceptance": 0.0076945293925521005,
  "telephone": 0.006687801202175559,
  "carderock division naval": 0.012617841748392759,
  "for navyrelated scientific": 0.013044037411284288,
  "bethesda maryland sponsor": 0.013044037411284288,
  "applications": 0.005806520386070833,
  "current work worksinprogress": 0.013044037411284288,
  "june for": 0.012617841748392759,
  "published in any": 0.013044037411284288,
  "mar gmtapr gmt": 0.014671611172274078
}
```