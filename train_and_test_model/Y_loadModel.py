
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys

def twoTupleDic():
    AA_list_sort = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N','P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'O']
    AA_dict = {}
    numm = 1
    for i in AA_list_sort:
        for j in AA_list_sort:
            AA_dict[i+j] = numm
            numm += 1
    return AA_dict
            

def DNA2Sentence(dna, K):
	sentence = ""
	length = len(dna)
	for i in range(length - K + 1):
		sentence += dna[i: i + K] + " "
	sentence = sentence[0 : len(sentence) - 1]
	return sentence

def Get_Unsupervised(fname,kmer):
    f = open(fname,'r')
    #g = open(gname,'w')
    seq = []
    lab = []
    k = kmer
    for i in f:
        if '>' not in i:
            i = i.strip('\n').upper()
            line = DNA2Sentence(i,k)
            seq.append(line)
        else:
            lab.append(i.strip('\n'))
    length = len(seq)
    f.close()
    return seq,length,lab


#load model
######################
fastafile = sys.argv[1]
model = sys.argv[2]
result = sys.argv[3]
#fastafile = 'seq.txt'
kmer = 2

texts,posnum,labels2= Get_Unsupervised(fastafile,kmer)

word_index = twoTupleDic()
sequences = []
for each in texts:
    each_index_list = []
    each = each.split(' ')
    for i in each:
        each_index_list.append(word_index[i])
    sequences.append(each_index_list)

MAX_SEQUENCE_LENGTH = 100
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = []
for i in range(0,posnum):
    labels.append(1)
for j in range(posnum,len(texts)):
    labels.append(0)

new_model = load_model(model)
loss, acc = new_model.evaluate(data, labels)
######################################### write
g = open(result,'w')
pre_test_y = new_model.predict(data)# prediction rate

for i in range(len(labels2)):
    g.write(labels2[i]+'\t'+str(pre_test_y[i][0])+'\n')

g.close()
