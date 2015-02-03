#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple
import bleu
import random
import numpy as np
import cPickle

optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default="100-best.list", help="N-best file")
optparser.add_option("-r", "--reference", dest="reference", default="/home/zhelunw/Documents/head500-ref.data", help="English reference sentences")
(opts, _) = optparser.parse_args()

num_features = 4

ref = [line.strip().split() for line in open(opts.reference)]#load reference file
candidate = namedtuple("candidate", "english, features , score")
nbests = []
cnt=0#count # of sentence
#we can run the first part for only once and save it. 
###1st part,compute blue score for each candidate translation.
for line in open(opts.nbest):    
    cnt = cnt + 1    
    #print '{0}\r'.format("\rIteration: %d/%d." %(cnt, 432303)),
    (i, sentence, features) = line.strip().split("|||")
    if len(nbests) <= int(i):
        nbests.append([])
    features = [float(h) for h in features.strip().split()]
    stats = [0 for kk in xrange(10)]#code from score-reranker.py
    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(sentence.strip().split(),ref[int(i)]))]
    score = bleu.smoothed_bleu(stats)
    nbests[int(i)].append( candidate(sentence.strip(), features, score)  )
cPickle.dump(nbests, open('my_nbests_add.p', 'wb'))#save the result. no need to run the first part each time
#print "finished calculating nbests."
nbests = cPickle.load(open('my_nbests_add.p', 'rb'))#load pickled file

#2nd part,learn the optimal weight
epochs = 20#setup parameters mentioned in pseudocode
tau_maxsize = 100#5000
xi =10#50
tau = []
alpha = 0.05
eta = 0.1
theta=[1.0/num_features for xxx in xrange(num_features)]#initialize weight
for epoch in range(epochs):
    #print '{0}\r'.format("\r nbest_index: %d/%d." %(epoch, epochs))
    for (index,nbest) in enumerate(nbests):
        #print '{0}\r'.format("\r nbest_index: %d/%d." %(index, len(nbests))),
        if len(tau) <= index:#code from rerank.py
            tau.append([])
        sample_size = len(tau[index])
        cnt_iter = 0
        while sample_size < tau_maxsize and cnt_iter < 10000:#sample nbest first 
            sample_size = len(tau[index])
            cnt_iter  = cnt_iter + 1#prevent dead loop, for some french sentences with few english candidates, we can not find enough samples
            rand1 = random.randint(0, len(nbest)-1)
            rand2 = random.randint(0, len(nbest)-1)
            if np.fabs(nbest[rand1].score - nbest[rand2].score )> alpha:
                if nbest[rand1].score > nbest[rand2].score:
                    tau[index].append((nbest[rand1],nbest[rand2]))
                else:
                    tau[index].append((nbest[rand2],nbest[rand1]))
            else:
                continue
        #print "out of loop",index
        tau[index] = sorted(tau[index],key=lambda h: h[0].score - h[1].score, reverse=True)[:xi]#prune to top xi
        tau_now = tau[index]
        for k in range(len(tau_now)) :#update process
            s1 = tau_now[k][0]
            s2 = tau_now[k][1]
            if np.dot(theta , s1.features) <= np.dot(theta , s2.features):
                #mistakes += 1
                f_diff = [s1.features[k] - s2.features[k] for k in range(len(s1.features)) ]
                theta += np.dot(eta , f_diff)#perceptron update

print "\n".join([str(weight) for weight in theta])
