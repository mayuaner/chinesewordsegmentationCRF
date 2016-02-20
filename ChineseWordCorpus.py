"""Yuanyuan Ma
LING131 Final Project
12/16/2015
"""

from corpus import Document
import csv
from glob import glob
import json
from os.path import basename, dirname, split, splitext


class Sentence(object):
    """this is to contain a sentence of the tagged sentences"""
    
    def __init__(self, document_list):
        self.sentence = document_list
        self.node = [x.node for x in document_list]

    def __len__(self): return len(self.sentence)
    def __iter__(self): return iter(self.sentence)
    def __getitem__(self, key): return self.sentence[key]
    def __setitem__(self, key, value): self.sentence[key] = value
    def __delitem__(self, key): del self.sentence[key]

class Feature(Document):
    """This is the feature that looks at three spaces in the Chinese
    word corpus"""

    def get_features(self, current, sentence):
        features = ['C0=%s' % sentence[current].data[0]] #current feature
        if current == 0: #the feature before current
            features.append('C-1=START')
        else:
            features.append('C-1=%s' % sentence[current-1].data[0])

        if current == (len(sentence)-1): #the feature after current
            features.append('C+1=END')
        else:
            features.append('C+1=%s' % sentence[current+1].data[0])
        return features

class Feature2(Document):
    """This is the feature designed for Chinese word segmentation.
    use features of five spaces"""

    #The feature is designed by me
    def get_features(self, current, sentence):
        features = ['C0=%s' % sentence[current].data[0]] #current feature
        for i in range(1, 2+1): #look at two features 
            if (current + i) >= len(sentence):
                features.append('C+%s=END' % i)
            else:
                features.append('C+%s=%s' % (i, sentence[current+i].data[0]))
            if (current - i) < 0:
                features.append('C-%s=END' % i)
            else:
                features.append('C-%s=%s' % (i, sentence[current-i].data[0]))
        return features
    
class Feature3(Document):
    """This is the feature designed for Chinese word segmentation.
    use features of seven spaces"""

    def get_features(self, current, sentence):
        features = ['C0=%s' % sentence[current].data[0]]
        for i in range(1, 3+1): #look at three spaces
            if (current + i) >= len(sentence):
                features.append('C+%s=END' % i)
            else:
                features.append('C+%s=%s' % (i, sentence[current+i].data[0]))

            if (current - i) < 0:
                features.append('C-%s=END' % i)
            else:
                features.append('C-%s=%s' % (i, sentence[current-i].data[0]))
        return features

class ChineseWordCorpus:
    """this is a class for Chinese word segmentation"""

    def __init__(self, datafiles, document_class=Document):
        self.documents = []
        self.datafiles = glob(datafiles)
        for datafile in self.datafiles:
            self.load(datafile, document_class)

    def __len__(self): return len(self.documents)
    def __iter__(self): return iter(self.documents)
    def __getitem__(self, key): return self.documents[key]
    def __setitem__(self, key, value): self.documents[key] = value
    def __delitem__(self, key): del self.documents[key]
    
    def load(self, datafile, feature_type=Feature):
        with open(datafile, 'r', encoding='utf8') as file:
            sentence = []
            ln = 0
            for line in file:
                character, tag = line.strip().split(' ')
                if character == '\u3002' or tag == 'O':
                    #to mark the end of a sentence and store it in the sentence object
                    if len(sentence) > 0:
                        self.documents.append(Sentence(sentence))
                    sentence = []
                else:
                    sentence.append(feature_type((character,), tag, ln))
                ln += 1
        self.get_feature()
        
    def get_feature(self):
        #map the features to a dictionary and use the integer for CRF purpose 
        self.nodedict = {}
        self.featuredict = {}
        for sentence in self.documents:
            for c, document in enumerate(sentence):
            #label the current node in the sentence, label with an integer 
                features = document.get_features(c, sentence)
                for feature in features:
                    if feature not in self.featuredict:
                        self.featuredict[feature] = len(self.featuredict)
                    document.fv.append(self.featuredict[feature])
                if document.node not in self.nodedict:
                   self.nodedict[document.node] = len(self.nodedict)
                document.node_i = self.nodedict[document.node]


class Seed(object):

    def __init__(self, path):
        self.path=path
        self.doclist = list(os.walk(path))[0][2]

    def get_seed_2tag(self, seed_number=3): #default to choose Peking University
        """this method transforms the training data from SIGHAN to 2-tag system files
        in this project, I use the training data from Peking University"""
        
        filepath = os.path.join(self.path, self.doclist[seed_number])
        file = open(filepath,'r', encoding = 'utf8')
        with open('seed_pku_2','w', newline = '') as fp:
            writer = csv.writer(fp, delimiter = ' ')
            for line in file:
                tl = []
                for i in range(len(line)):
                    if line[i]!= ' ' and line[i] != '\n':
                        if line[i] == '\u3002':
                            tl.append((line[i], 'O'))
                        elif i == 0 or line[i-1] == '\u3000' or line[i-1] == '\n':
                            tl.append((line[i], 'B'))
                        else:
                            tl.append((line[i], 'I'))
                writer.writerows(tl)

    def get_seed_4tag(self, seed_number = 3):   #default to choose Peking University
        """this method transforms the training data from SIGHAN to 4-tag system
        data from Peking University"""
        filepath = os.path.join(self.path, self.doclist[seed_number])
        file = open(filepath,'r', encoding = 'utf8')
        with open('seed_pku_4','w', newline = '') as fp:
            writer = csv.writer(fp, delimiter = ' ')
            for line in file:
                tl = []
                for i in range(len(line)):
                    if line[i]!= ' ' and line[i] != '\n':
                        if line[i] == '\u3002':
                            tl.append((line[i], 'O'))
                        elif (i == 0 or line[i-1] == '\u3000' or line[i-1] == '\n') and (line[i+1] == '\u3000' or line[i+1] == '\n'):
                            tl.append((line[i], 'S'))
                        elif i == 0 or line[i-1] == '\u3000' or line[i-1] == '\n':
                            tl.append((line[i], 'B'))
                        elif line[i+1] == '\u3000' or line[i+1] == '\n':
                            tl.append((line[i], 'E'))
                        else:
                            tl.append((line[i], 'M'))
                writer.writerows(tl)
    
