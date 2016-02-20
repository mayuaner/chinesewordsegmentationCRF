# chinesewordsegmentationCRF
----------------------------------
--------------------------------------
CRF-Chinese Word Segmentation Training 
--------------------------------------

Author: Yuanyuan Ma

Package Used: 
A CRF algorithm package:
corpus.py
crf.py
word_segmenter.py

Training data:
Peking University Corpus

====================================
How to run?
====================================
run the word_segmenter.py in IDLE.
You can specify the parameter as "seed_pku.small", and replace it 
with other training files.
Also, replace 'Feature' with 'Feature2' or 'Feature3' can give out different 
result.

It will print the accuracy rate on each batch of training set.

-------------------------
Design Decisions & Issues
-------------------------

Features used in the training model:

Character: 
look at the current node tag and a previous one tag and a tag afterwards.
for example:
希 B
望 I
的 B
for the words "望", it look at one previous "希" and one afterwards "的"

Character2:
looks at the current node and two previous and two afterward nodes.
for example:
满 I
希 B
望 I
的 B
新 B
for the character "望", it looks a space of five

Character3:
likewise, this character uses 7 spaces.

====================
Class and Functions
====================

def get_training()
this methods takes in a training data and output a "seed" for the CRF package. 
The tagging system I use is a binary tagging system.

"B" indicates the beginning of a words and "I" indicates the middle or last character.
"O" indicates the end of a sentence.

class Character(Document)
This class has one function to get the features. 
The features are stored in a list, marked by current node C and c-1 and c+1

class ChineseWordCorpus:
load() method takes in the "seed" file.
Each sentence is stored as a Sequence.
Each line of the file is in a Document class (which is also a Character)

get_feature() method maps the features to a dictionary and stores in it.
It also passes a vector to the CRF function to do the calculation.

-----------------------------------
Segmentation Result and Discussion
-----------------------------------
The eventual accuracy rate of Character (using 3 feature space, and one space looking-at-environment) 
gets a result around 86%, Character2 (5 spaces) gets a rate of 85% and Character3 (7 Spaces) gets 
a result of 85%.

It it insteresting more spaces actually yields to a lower rate of accuracy.

Further improvement can be done using a more sophisticated tagging system.
Instead of 2-tag system, it is possible to use a 4-tag system. 

The matrix multiplication method can be implemented using logsumexp() method. Specific
implementation needs to be further looked into.

---------------------------------
update 12/16/2015
--------------------------------
implemented 4-tag system, overflown error has occurred again.

