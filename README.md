# DeepIPs
Identification of phosphorylation sites using deep-learning method


Identification of phosphorylation sites in host cells infected with SARS-CoV-2 using CNN-LSTM architecture.

1. File 'train_and_test_model' contains scripts for training the model and loading model. The Glove_CNNLSTM.py, SEL_CNNLSTM.py, fasttext_CNNLSTM.py, and word2vec_CNNLSTM.py are scripts based on Glove, SEL, fastText, and word2vec word embedding methods, respectively. ST_loadModel.py and Y_loadModel.py were used to predict S/T phosphorylation sites and Y phosphorylation sites, respectively.

2. File 'model' contains models trained by optimal word embedding method and CNN-LSTM architecture. ST_model.h5 and Y_model.h5 are final model files based on S/T phosphorylation sites and Y phosphorylation sites, respectively.

3. File 'data' contains training and testing data used in this study. ST-train.fa and ST-test.fa were used to train and test S/T sites based model; likewise, Y-train.fa and Y-test fa were used to train and test Y sites based model.

We encourage users to use our online web-server at http://lin-group.cn/server/DeepIPs.
