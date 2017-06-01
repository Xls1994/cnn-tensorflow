from utils import *
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from cnn_train import *
import cPickle
#ftrain = "data/combine.train"
#ftrain = "data/temp.train"
ftrain = 'data/beth.train'

path ='data.pkl'
# saveData(ftrain,path)
W_train, P_train, d1_train, d2_train, T_train, Att1_train, Att2_train,Y_train,seq_len, att1_len, att2_len, label_dict_size, word_dict_size,pos_dict_size, d1_dict_size, d2_dict_size, type_dict_size\
        =cPickle.load(open(path,'rb'))


fp1 = open("accuracy_att_2.dat","w")
fp2 = open("f1_att_2.dat","w")
acc_list = []
f1_list = []
kf = KFold(len(W_train), n_folds=2)
W_tr, W_te,P_tr, P_te,d1_tr, d1_te,d2_tr, d2_te,T_tr, T_te ,Att1_tr, Att1_te,Att2_tr, Att2_te,Y_tr, Y_te  =train_test_split(W_train,P_train,d1_train,
                                                     d2_train,T_train,Att1_train,
                                                     Att2_train,Y_train,
                                    test_size=0.2,random_state=1377)


model = CNN_Train(label_dict_size,
        seq_len, 			#length of largest sent
        att1_len,			#length of largest entity 1
        att2_len,			#length of largest  entity 2
        label_dict_size, 	#number of classes
        word_dict_size,		#word vocab length
        pos_dict_size,		#pos vocab length
        d1_dict_size,		#d1 vocab length
        d2_dict_size,		#d2 vocab length
        type_dict_size)		#type vocab length

print W_tr
acc,f1_score = model.cnnTrain(W_tr, W_te, P_tr, P_te, d1_tr, d1_te, d2_tr, d2_te, T_tr, T_te, Att1_tr, Att1_te, Att2_tr, Att2_te, Y_tr, Y_te)

print "Accuracy = ", acc
print "F1 score = ", f1_score
acc_list.append(acc)
f1_list.append(f1_score)
fp1.write("%f\n"%acc)
fp2.write("%f\n"%f1_score)

# for train, test in kf:
#     W_tr, W_te = W_train[train], W_train[test]
#     P_tr, P_te = P_train[train], P_train[test]
#     d1_tr, d1_te = d1_train[train], d1_train[test]
#     d2_tr, d2_te = d2_train[train], d2_train[test]
#     T_tr, T_te = T_train[train], T_train[test]
#     Att1_tr, Att1_te = Att1_train[train], Att1_train[test]
#     Att2_tr, Att2_te = Att2_train[train], Att2_train[test]
#     Y_tr, Y_te = Y_train[train], Y_train[test]
#
#     # print W_tr.shape, W_te.shape
#     # print P_tr.shape, P_te.shape
#     # print d1_tr.shape, d1_te.shape
#     # print d2_tr.shape, d2_te.shape
#     # print T_tr.shape, T_te.shape
#     # print Y_tr.shape, Y_te.shape
#
#
#     model = CNN_Train(label_dict_size,
#             seq_len, 			#length of largest sent
#             att1_len,			#length of largest entity 1
#             att2_len,			#length of largest  entity 2
#             label_dict_size, 	#number of classes
#             word_dict_size,		#word vocab length
#             pos_dict_size,		#pos vocab length
#             d1_dict_size,		#d1 vocab length
#             d2_dict_size,		#d2 vocab length
#             type_dict_size)		#type vocab length
#
#     print W_tr
#     acc,f1_score = model.cnnTrain(W_tr, W_te, P_tr, P_te, d1_tr, d1_te, d2_tr, d2_te, T_tr, T_te, Att1_tr, Att1_te, Att2_tr, Att2_te, Y_tr, Y_te)
#
#     print "Accuracy = ", acc
#     print "F1 score = ", f1_score
#     acc_list.append(acc)
#     f1_list.append(f1_score)
#     fp1.write("%f\n"%acc)
#     fp2.write("%f\n"%f1_score)
#
# print "Average accuracy = ", np.mean(acc_list)
# fp1.write("Average=%f"%np.mean(acc_list))
# print "Average F1 score = ", np.mean(f1_list)
# fp2.write("Average=%f"%np.mean(f1_list))