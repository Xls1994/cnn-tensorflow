# from helper import *
import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib.tensorboard.plugins import projector
class CNN_Relation(object):		 
			 
    def __init__(self, seq_len, att1_len, att2_len,
        num_classes, vocab_size, pos_dict_size, p1_dict_size, p2_dict_size, type_dict_size,
        w_emb_size, p1_emb_size, p2_emb_size, pos_emb_size, type_emb_size,
        filter_sizes, num_filters,l2_reg_lambda = 0.0,batch_size=64):

        emb_size = w_emb_size + p1_emb_size + p2_emb_size + pos_emb_size + type_emb_size
        print 'embedding size:',emb_size

        # set data
        with tf.name_scope('input'):
            self.x  = tf.placeholder(tf.int32, [batch_size, seq_len], name="x")
            self.x1 = tf.placeholder(tf.int32, [batch_size, seq_len], name="x1")
            self.x2 = tf.placeholder(tf.int32, [batch_size, seq_len], name="x2")
            self.x3 = tf.placeholder(tf.int32, [batch_size, seq_len], name="x3")
            self.x4 = tf.placeholder(tf.int32, [batch_size, seq_len], name='x4')
            self.att1 = tf.placeholder(tf.int32, [batch_size, att1_len], name="att1")
            self.att2 = tf.placeholder(tf.int32, [batch_size, att2_len], name="att2")

            self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # initial the embedding
        W_wemb =  tf.Variable(tf.random_uniform([vocab_size, w_emb_size], -1.0, +1.0))
        W_p1emb = tf.Variable(tf.random_uniform([p1_dict_size, p1_emb_size], -1.0, +1.0))
        W_p2emb = tf.Variable(tf.random_uniform([p2_dict_size, p2_emb_size], -1.0, +1.0))
        W_posemb = tf.Variable(tf.random_uniform([pos_dict_size, pos_emb_size], -1.0, +1.0))
        W_temb = tf.Variable(tf.random_uniform([type_dict_size, type_emb_size], -1.0, +1.0))

        #Embedding layer
        emb_ent1 = tf.nn.embedding_lookup(W_wemb, self.att1)	#entity 1 embedding
        emb_ent2 = tf.nn.embedding_lookup(W_wemb, self.att2)	#entity 2 embedding
        emb = tf.nn.embedding_lookup(W_wemb, self.x)			#word embedding
        emb1 = tf.nn.embedding_lookup(W_p1emb, self.x1)			#position from first entity embedding
        emb2 = tf.nn.embedding_lookup(W_p2emb, self.x2)			#position from second entity embedding
        emb3 = tf.nn.embedding_lookup(W_posemb, self.x3)		#POS embedding
        emb4 = tf.nn.embedding_lookup(W_temb, self.x4)		    #Type embedding

        #Attention layer
        a1 = tf.unstack(emb_ent1,axis=0)
        a2 = tf.unstack(emb_ent2,axis=0)
        a = tf.unstack(emb,axis=0)
        r1 = []
        r2 = []
        r1_mean = []
        r2_mean = []
        with tf.name_scope('attention_entity_input'):
            for i in range(0,batch_size):
                r1.append(tf.matmul(a1[i],a[i],transpose_b=True))   #shape (att1_len,seq_len)
                r2.append(tf.matmul(a2[i],a[i],transpose_b=True))   #shape (att2_len,seq_len)

            for i in range(0,batch_size):
                r1_mean.append(tf.reduce_max(r1[i],axis=0,name=None))	# shape(seq_len,)
                # Max of inner products with each word in the entity 1
                r2_mean.append(tf.reduce_max(r2[i],axis=0,name=None))	# shape(seq_len,)
                # Max of inner products with each word in the entity 2

            r1 = []
            r2 = []
            r = []
            for i in range(0,batch_size):
                #shape (seq_len,w_emb_size)
                r1.append(tf.matmul(tf.matrix_set_diag(tf.zeros([seq_len,seq_len],tf.float32),r1_mean[i]),a[i]))
                r2.append(tf.matmul(tf.matrix_set_diag(tf.zeros([seq_len,seq_len],tf.float32),r2_mean[i]),a[i]))
                r.append(tf.scalar_mul(0.5,tf.add(r1[i],r2[i])))
            #shape (bathch_size,seq_len,w_emb_size)
            r = tf.stack(r)

            # print (r.get_shape())
            # print (emb.get_shape())

            X = tf.concat(axis=2, values=[r, emb1, emb2, emb3, emb4])			#shape(?, 98, 70)

            X_expanded = tf.expand_dims(X, -1) 				#shape (?, 98, 70, 1)

        l2_loss = tf.constant(0.0)

        #CNN+Maxpooling Layer
        pooled_outputs = []
        with tf.name_scope('conv_layer'):
            for i, filter_size in enumerate(filter_sizes):
                filter_shape = [filter_size, emb_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                        X_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 	#shape (?, 19, 1, 70)
            # print "h ", h.get_shape
            # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
            # print "pooled", pooled.get_shape			#shape=(?, 1, 1, 70)

                pooled_outputs.append(pooled)

        print ("pooled_outputs", len(pooled_outputs))

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(axis=3, values=pooled_outputs)				#shape= (?, 1, 1, 210)
        print ("h_pool", h_pool.get_shape)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])	#shape =(?, 210)
        print ("h_pool_flat", h_pool_flat.get_shape)
        with tf.name_scope('dropout_layer'):
        #dropout layer
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope('fully_connected_layer'):
        #Fully connected layer
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")

        #prediction and loss function
        self.predictions = tf.argmax(scores, 1, name="predictions")
        with tf.name_scope('losses'):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss
            tf.summary.scalar('cross_entropy', self.loss)

        # Accuracy
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_predictions'):
                self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
                tf.summary.scalar('accuracy',self.accuracy)

        self.merged  = tf.summary.merge_all()
        # self.sess=tf.Session()
        # self.summary_writer =tf.summary.FileWriter('./logs', self.sess.graph)
        # config = projector.ProjectorConfig()
        # embedding_conf = config.embeddings.add()
        # embedding_conf.tensor_name = W_wemb.name
        # embedding_conf.metadata_path ='./logs/metadata.tsv'
        # projector.visualize_embeddings(self.summary_writer, config)