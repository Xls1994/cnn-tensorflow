from cnn_text import CNN_Relation
import numpy as np
import tensorflow as tf
import random
import sklearn as sk


class CNN_Train(object):

    def __init__(self,num_classes, seq_len, att1_len, att2_len,
                 label_dict_size, word_dict_size, pos_dict_size,
                 d1_dict_size, d2_dict_size, type_dict_size,
                 w_emb_size=50, d1_emb_size=5, d2_emb_size=5,
                 pos_emb_size=5, type_emb_size=5, filter_sizes=[2,3,5],
                 num_filters=70,batch_size=64):


        self.cnn = CNN_Relation(
            seq_len = seq_len,
            att1_len = att1_len,
            att2_len = att2_len,
            num_classes = num_classes,
            vocab_size = word_dict_size,
            pos_dict_size = pos_dict_size,
            p1_dict_size = d1_dict_size,
            p2_dict_size = d2_dict_size,
            type_dict_size = type_dict_size,
            w_emb_size = w_emb_size,
            p1_emb_size = d1_emb_size,
            p2_emb_size = d2_emb_size,
            pos_emb_size = pos_emb_size,
            type_emb_size = type_emb_size,
            filter_sizes =  filter_sizes,
            num_filters = num_filters,
            l2_reg_lambda = 0.0,
            batch_size=batch_size
            )
        self.batch_size =batch_size
        self.sess = tf.Session()

        self.optimizer = tf.train.AdamOptimizer(1e-2)

        self.grads_and_vars = self.optimizer.compute_gradients(self.cnn.loss)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        self.writer =tf.summary.FileWriter('./logs', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.usingWriter =True
            # self.fp = open("result.txt",'w')
		
    def train_step(self, W_batch, d1_batch, d2_batch, P_batch, T_batch, Att1_batch, Att2_batch, y_batch):
        feed_dict = {
            self.cnn.x :W_batch,
            self.cnn.x1:d1_batch,
            self.cnn.x2:d2_batch,
            self.cnn.x3:P_batch,
            self.cnn.x4:T_batch,
            self.cnn.att1:Att1_batch,
            self.cnn.att2:Att2_batch,
            self.cnn.input_y:y_batch,
            self.cnn.dropout_keep_prob: 0.5
            }
        if self.usingWriter:
            _, step, loss, accuracy, predictions,summary = self.sess.run(
                [self.train_op, self.global_step, self.cnn.loss,
                 self.cnn.accuracy, self.cnn.predictions,self.cnn.merged],
                feed_dict)
            self.writer.add_summary(summary, step)
        else:
            _, step, loss, accuracy, predictions = self.sess.run(
                [self.train_op, self.global_step, self.cnn.loss,
                 self.cnn.accuracy, self.cnn.predictions],
                feed_dict)
        print ("step "+str(step) + " loss "+str(loss) +" accuracy "+str(accuracy))



    def test_step(self, W_batch, d1_batch, d2_batch, P_batch, T_batch, Att1_batch, Att2_batch, y_batch):
        feed_dict = {
            self.cnn.x :W_batch,
            self.cnn.x1:d1_batch,
            self.cnn.x2:d2_batch,
            self.cnn.x3:P_batch,
            self.cnn.x4:T_batch,
            self.cnn.att1:Att1_batch,
            self.cnn.att2:Att2_batch,
            self.cnn.input_y:y_batch,
            self.cnn.dropout_keep_prob:1.0
            }
        step, loss, accuracy, predictions = self.sess.run([self.global_step, self.cnn.loss, self.cnn.accuracy, self.cnn.predictions], feed_dict)
        print ("Accuracy in test data", accuracy)
        y_pred = predictions
        y_true = np.argmax(y_batch,1)
        # f1_score = 0
        f1_score = sk.metrics.f1_score(y_true,y_pred,pos_label=None,average='weighted')
            # print ()
        return accuracy,f1_score

    def cnnTrain(self, W_tr, W_te, P_tr, P_te, d1_tr, d1_te, d2_tr, d2_te, T_tr, T_te, Att1_tr, Att1_te, Att2_tr, Att2_te, Y_tr, Y_te):
        print (len(P_te))

        time = range(len(W_tr))
        step = np.random.shuffle(time)
        batch_size=self.batch_size
        j = 0
        for i in range(10):
            if(j >= len(W_tr)-batch_size):
                j=0
    #			self.train_step(W_tr[step[j]], d1_tr[step[j]], d2_tr[step[j]], P_tr[step[j]], T_tr[step[j]], Y_tr[step[j]])
            s = range(j, j+batch_size)
            self.train_step(W_tr[s], d1_tr[s], d2_tr[s], P_tr[s], T_tr[s], Att1_tr[s], Att2_tr[s], Y_tr[s])
            j += batch_size

        j = 0
        acc = []
        f1 = []
        for i in range(26):
            if(j >= len(W_te)-batch_size):
                j=0
            s = range(j, j+batch_size)
            acc_cur,f1_score = self.test_step(W_te[s], d1_te[s], d2_te[s], P_te[s], T_te[s], Att1_te[s], Att2_te[s], Y_te[s])
            acc.append(acc_cur)
            f1.append(f1_score)
            j += batch_size

        acc = np.mean(acc)
        f1 = np.mean(f1)
        self.sess.close()

        # y_true = np.argmax(Y_te, 1)
        # y_pred = pred
    #		print "Precision", sk.metrics.precision_score(y_true, y_pred, average=None )
    #   	print "Recall", sk.metrics.recall_score(y_true, y_pred, average=None )
        # f1_score = sk.metrics.f1_score(y_true, y_pred, average='weighted')
    #    	print "confusion_matrix"
    #   	print sk.metrics.confusion_matrix(y_true, y_pred)
        # self.fp.write(sk.metrics.f1_score(y_true, y_pred, average=None))
        # self.fp.write("%f\n"%acc)

        """
        for t,p in zip(y_true, y_pred):
        fp.write(str(t) +" "+str(p))
        fp.write('\n')
        fp.close()
        """
        return acc,f1


