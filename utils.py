import re
import numpy as np
import nltk
import cPickle



def readData(ftrain):
  fp = open(ftrain, 'r')
  samples = fp.read().strip().split('\n\n\n')
  sent_names     = []        #1-d array 
  sent_lengths   = []        #1-d array
  sent_contents  = []        #2-d array [[w1,w2,....] ...]
  sent_lables    = []        #1-d array
  entity1_list   = []        #2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
  entity2_list   = []        #2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
  for sample in samples:
    name, sent, entities, relation = sample.strip().split('\n')    
    sent_lengths.append(len(sent.split()))
    sent_names.append(name)
    sent_contents.append(sent.lower().split())
    
    m = re.match(r"\(\[['\"](.*)['\"], (\d*), (\d*), '(.*)'\], \[['\"](.*)['\"], (\d*), (\d*), '(.*)'\]\)", entities.strip())
    if m :
        e1   = m.group(1)
        e1_s = int(m.group(2))
        e1_e = int(m.group(3))
        e1_t = m.group(4)

        e2   = m.group(5)
        e2_s = int(m.group(6))
        e2_e = int(m.group(7))
        e2_t = m.group(8)
        if(e1_s < e2_s):
            entity1_list.append([e1,e1_s,e1_e,e1_t])
            entity2_list.append([e2,e2_s,e2_e,e2_t])
        else:
            entity1_list.append([e2,e2_s,e2_e,e2_t])
            entity2_list.append([e1,e1_s,e1_e,e1_t])
#        print e1,e2
    else:
        print ("Error in reading", entities.strip())
#        exit(0)
    
    ma = re.match(r"\[['\"](.*)['\"], '(.*)', ['\"](.*)['\"]\]", relation.strip())
    if(ma):
        lable = ma.group(2)        
    elif relation == '[0]':
        lable = 'other'
    else:
        print ("Error in reading", relation)
        exit(0)
#    print lable
    sent_lables.append(lable)
  return sent_contents,entity1_list, entity2_list, sent_lables 

def makePosFeatures(sent_contents):
    pos_tag_list = []
    for sent in sent_contents:
#        tags = tagger.parse(sent)
#        sent_t, sent_o, sent_pos, sent_chunk, sent_bio = zip(*tags)
        pos_tag = nltk.pos_tag(sent)
        # print pos_tag
        pos_tag = zip(*pos_tag)[1]
        # print pos_tag
        pos_tag_list.append(pos_tag)        
    return pos_tag_list 

def mapWordToId(sent_contents, word_dict):
    T = []
    for sent in sent_contents:
        t = []
        for w in sent:
            if w in word_dict:
                t.append(word_dict[w])
            else:
                t.append(word_dict['unknown'])
        T.append(t)
    return T

def makeDistanceFeatures(sent_contents, entity1_list, entity2_list):
    d1_list = []
    d2_list = []
    type_list = []
    for sent, e1_part, e2_part in zip(sent_contents, entity1_list, entity2_list):
        entity1, s1, e1, t1 = e1_part
        entity1, s2, e2, t2 = e2_part
        maxl = len(sent)

        d1 = []        
        for i in range(maxl):
            if i < s1 :
                d1.append(str(i - s1))
            elif i > e1 :
                d1.append(str(i - e1 ))
            else:
                d1.append('0')
        d1_list.append(d1)

        d2 = []
        for i in range(maxl):
            if i < s2 :
                d2.append(str(i - s2))
            elif i > s2 :
                d2.append(str(i - s2))
            else:
                d2.append('0')        
        d2_list.append(d2)
        
        t = []
        for i in range(maxl):
            t.append('Out')
        for i in range(s1,e1+1):
            if(t1 == 'problem'):
                t[i] = 'Prob'
            elif(t1 == 'treatment'):
                t[i] = 'Treat'
            elif(t1 == 'test'):
                t[i] = 'Test'

        for i in range(s2, e2+1):
            if(t2 == 'problem'):
                t[i] = 'Prob'
            elif(t2 == 'treatment'):
                t[i] = 'Treat'
            elif(t2 == 'test'):
                t[i] = 'Test'
        type_list.append(t)
    return d1_list, d2_list, type_list

def makeAttentionFeatures(entity1_list,entity2_list):
    att1_list = []
    att2_list = []

    for i in entity1_list:
        att1_list.append(i[0])

    for i in entity2_list:
        att2_list.append(i[0])

    return att1_list,att2_list

def makeWordList(sent_list):
    wf = {}
    for sent in sent_list:
        for w in sent:
            if w in wf:
                wf[w] += 1
            else:
                wf[w] = 1
    wl = {}
    i = 0
    wl['unknown'] = 0    
    for w,f in wf.iteritems():
        wl[w] = i
        i += 1 
    return wl

def makeRelList(rel_list):
    rel_dict = {}
    for rel in rel_list:
        if rel in rel_dict:
            rel_dict[rel] += 1
        else:
            rel_dict[rel] = 0
    wl = {}
    i = 0
    for w,f in rel_dict.iteritems():
        wl[w] = i
        i += 1
    return wl 

def makePaddedList(sent_contents, pad_symbol= '<pad>'):
    maxl = max([len(sent) for sent in sent_contents])
    T = []
    for sent in sent_contents:
        t = []
        lenth = len(sent)
        for i in range(lenth):
            t.append(sent[i])
        for i in range(lenth,maxl):
            t.append(pad_symbol)
        T.append(t)

    return T, maxl

def mapLabelToId(sent_lables, label_dict):
    return [label_dict[label] for label in sent_lables]

def saveData(trainPath='data/beth.train',mrPath='data/data.pkl'):

    sent_contents, entity1_list, entity2_list, sent_lables = readData(trainPath)

    #Featurizer
    pos_tag_list = makePosFeatures(sent_contents)
    d1_list, d2_list, type_list = makeDistanceFeatures(sent_contents, entity1_list, entity2_list)
    att1_list, att2_list = makeAttentionFeatures(entity1_list,entity2_list)

    #padding
    sent_contents,seq_len = makePaddedList(sent_contents)
    att1_list,att1_len = makePaddedList(att1_list)
    att2_list,att2_len = makePaddedList(att2_list)
    pos_tag_list,_ = makePaddedList(pos_tag_list)
    d1_list,_ = makePaddedList(d1_list)
    d2_list,_ = makePaddedList(d2_list)
    type_list,_ = makePaddedList(type_list)

    # Wordlist
    word_dict = makeWordList(sent_contents)
    pos_dict = makeWordList(pos_tag_list)
    d1_dict = makeWordList(d1_list)
    #print "d1_dict", d1_dict
    d2_dict = makeWordList(d2_list)
    type_dict = makeWordList(type_list)

    #label_dict = makeRelList(sent_lables)
    label_dict = {'other':0, 'TrWP': 1, 'TeCP': 2, 'TrCP': 3, 'TrNAP': 4, 'TrAP': 5, 'PIP': 6, 'TrIP': 7, 'TeRP': 8}

    #print (label_dict)


    #vocabulary size
    word_dict_size = len(word_dict)
    pos_dict_size = len(pos_dict)
    d1_dict_size = len(d1_dict)
    d2_dict_size = len(d2_dict)
    type_dict_size = len(type_dict)
    label_dict_size = len(label_dict)

    #print "pos dict", pos_dict
    # Mapping
    W_train =  np.array(mapWordToId(sent_contents, word_dict))
    P_train = np.array(mapWordToId(pos_tag_list, pos_dict))
    d1_train = np.array(mapWordToId(d1_list, d1_dict))
    d2_train = np.array(mapWordToId(d2_list, d2_dict))
    T_train = np.array(mapWordToId(type_list,type_dict))
    Att1_train = np.array(mapWordToId(att1_list,word_dict))
    Att2_train = np.array(mapWordToId(att2_list,word_dict))
    print Att1_train.shape

    Y_t = mapLabelToId(sent_lables, label_dict)
    Y_train = np.zeros((len(Y_t),label_dict_size))
    for i in range(len(Y_t)):
        Y_train[i][Y_t[i]] = 1.0

    cPickle.dump([W_train, P_train, d1_train, d2_train, T_train,Att1_train,Att2_train,Y_train,
                  seq_len,att1_len,att2_len,label_dict_size,word_dict_size,
                  pos_dict_size,d1_dict_size,d2_dict_size,type_dict_size
                  ], open(mrPath, "wb"))

def loadData(path):
    W_train, P_train, d1_train, d2_train, T_train, Att1_train, Att2_train,Y_train,seq_len, att1_len, att2_len, label_dict_size, word_dict_size,pos_dict_size, d1_dict_size, d2_dict_size, type_dict_size\
        =cPickle.load(open(path,'rb'))

    print Att1_train
    print Att1_train.shape
    print W_train.shape
    print seq_len

if __name__=='__main__':
    sent_contents, entity1_list, entity2_list, sent_lables = readData('./data/beth.train')


    #Featurizer
    pos_tag_list = makePosFeatures(sent_contents)
    d1_list, d2_list, type_list = makeDistanceFeatures(sent_contents, entity1_list, entity2_list)
    att1_list, att2_list = makeAttentionFeatures(entity1_list,entity2_list)

    #padding
    sent_contents,seq_len = makePaddedList(sent_contents)
    att1_list,att1_len = makePaddedList(att1_list)
    att2_list,att2_len = makePaddedList(att2_list)
    pos_tag_list,_ = makePaddedList(pos_tag_list)
    d1_list,_ = makePaddedList(d1_list)
    d2_list,_ = makePaddedList(d2_list)
    type_list,_ = makePaddedList(type_list)

    # Wordlist
    word_dict = makeWordList(sent_contents)
    pos_dict = makeWordList(pos_tag_list)
    d1_dict = makeWordList(d1_list)
    #print "d1_dict", d1_dict
    d2_dict = makeWordList(d2_list)
    type_dict = makeWordList(type_list)
    Att1_train = np.array(mapWordToId(att1_list, word_dict))
    print att1_list
    print att1_len
    print word_dict
    print Att1_train
    pass
