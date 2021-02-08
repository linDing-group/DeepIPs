from __future__ import print_function
import _pickle as cPickle
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

import h5py
from keras.models import model_from_json
from keras.models import load_model
from sklearn.metrics import confusion_matrix

from sklearn import metrics

def trans(str1):
    a = []
    dic = {'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    return a

def createTrainData(str1):
    sequence_num = []
    label_num = []
    f = open(str1).readlines()
    for i in range(0,len(f)-1,2):
        label = f[i].strip('\n').replace('>','')
        label_num.append(int(label))
        sequence = f[i+1].strip('\n')
        sequence_num.append(trans(sequence))

    return sequence_num,label_num

def createTrainTestData(str_path, nb_words=None, skip_top=0,
              maxlen=None, test_split=0.25, seed=113,
              start_char=1, oov_char=2, index_from=3):
    X,labels = cPickle.load(open(str_path, "rb"))

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)
    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                                      'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])


    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    return (X_train, y_train), (X_test, y_test)


fastafile = 'Y-train.fa'
modelfile = "Y_vectors2_100.txt"
pklfile = "data1.pkl"
model_save = 'Y_fasttext_model.h5'


max_features = 23
maxlen = 200
embedding_size = 128

# Convolution
#filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 70

# Training
batch_size = 128
nb_epoch = 11

a,b = createTrainData(fastafile)
t = (a, b)
cPickle.dump(t,open(pklfile,"wb"))

(X_train, y_train), (X_test, y_test) = createTrainTestData(pklfile,nb_words=max_features, test_split=0.3)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

######################################################## k-fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
acc_score = []
auc_score = []
sn_score = []
sp_score = []
mcc_score = []

for i,(train, test) in enumerate(kfold.split(X_train, y_train)):
    print('\n\n%d'%i)

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=10,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=5,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))

    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')

    #早停法
    checkpoint = EarlyStopping(monitor='val_loss', 
               min_delta=0, 
               patience=3, 
               verbose=1, mode='auto')


    #model.fit(X_train[train], y_train[train], epochs = nb_epoch, batch_size = batch_size, validation_data = (X_train[test], y_train[test]), shuffle = True, callbacks=[checkpoint],verbose=2)
    model.fit(X_train[train], y_train[train], epochs = nb_epoch, batch_size = batch_size, validation_data = (X_train[test], y_train[test]), shuffle = True)
    ##########################
    prd_acc = model.predict(X_train[test])
    pre_acc2 = []
    for i in prd_acc:
        pre_acc2.append(i[0])

    prd_lable =[]
    for i in pre_acc2:
        if i>0.5:
             prd_lable.append(1)
        else:
             prd_lable.append(0)
    prd_lable = np.array( prd_lable)
    obj = confusion_matrix(y_train[test], prd_lable)
    tp = obj[0][0]
    fn = obj[0][1]
    fp = obj[1][0]
    tn = obj[1][1]
    sn = tp/(tp+fn)
    sp = tn/(tn+fp)
    mcc= (tp*tn-fp*fn)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
    sn_score.append(sn)
    sp_score.append(sp)
    mcc_score.append(mcc)
    ###########################
    pre_test_y = model.predict(X_train[test], batch_size = batch_size)
    test_auc = metrics.roc_auc_score(y_train[test], pre_test_y)
    auc_score.append(test_auc)
    print("test_auc: ", test_auc) 

    score, acc = model.evaluate(X_train[test], y_train[test], batch_size=batch_size)
    acc_score.append(acc)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('***********************************************************************\n')
print('***********************print final result*****************************')
print(acc_score,auc_score)
mean_acc = np.mean(acc_score)
mean_auc = np.mean(auc_score)
mean_sn = np.mean(sn_score)
mean_sp = np.mean(sp_score)
mean_mcc = np.mean(mcc_score)
#print('mean acc:%f\tmean auc:%f'%(mean_acc,mean_auc))

line = 'acc\tsn\tsp\tmcc\tauc:\n%.2f\t%.2f\t%.2f\t%.4f\t%.4f'%(100*mean_acc,100*mean_sn,100*mean_sp,mean_mcc,mean_auc)
print('5-fold result\n'+line)


#print('***************************save model********************************************')
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))#input_length是输入维度长度，embedding_size是每个向量经过嵌入层后生成的长度
model.add(Dropout(0.5))
# #nb_filter : 卷积核的数量，也是输出的维度。filter_length : 每个过滤器的长度。
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=10,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=5,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))

model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# #optimizer：优化器，如Adam；loss：计算损失，这里用的是交叉熵损失；metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。如果要在多输出模型中为不同的输出指定不同的指标，可向该参数传递一个字典，例如metrics={‘output_a’: ‘accuracy’}
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs = nb_epoch, batch_size = batch_size, validation_data = (X_test, y_test), shuffle = True)
model.save(model_save,overwrite=True)
