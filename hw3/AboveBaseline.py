#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
from math import log10

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training Start...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
f_count = defaultdict(int)
e_count = defaultdict(int)

# model 0: given e align f  model 1: given f align e
model = 1  
num_Null=3


for (n, (f, e)) in enumerate(bitext):
  if model == 0:
    for x in range(num_Null):
      e.insert(0,"NULL") 
  else:
    for x in xrange(num_Null):
      f.insert(0,"NULL")
  for f_i in set(f):
    f_count[f_i] += 1
  for e_j in set(e):
    e_count[e_j] += 1
  if n % 500 == 0:
    sys.stderr.write(".")

V_f=0
for f_i in f_count.keys():
  V_f+=f_count[f_i]

V_e=0
for e_j in e_count.keys():
  V_e+=e_count[e_j]


t_f_e = defaultdict(float)
sys.stderr.write("Start initialize\n")



for (n, (f, e)) in enumerate(bitext):
  for e_j in set(e):
    for f_i in set(f):
      if model == 0:
        t_f_e[(f_i,e_j)]=1.0/V_f
      else:
        t_f_e[(f_i,e_j)]=1.0/V_e
  if n % 500 == 0:
    sys.stderr.write(".")

sys.stderr.write("Finish initialize\n")

k=0
theta = 0.0005

while 1:
  if model == 0:
    sys.stderr.write("start EM iter:"+str(k)+"\n")
    k+=1
    fe_count = defaultdict(float)
    count_e = defaultdict(float)
    for (n, (f, e)) in enumerate(bitext):
      for f_i in set(f):
        Z=0
        for e_j in set(e):
          Z+=t_f_e[(f_i,e_j)]
        for e_j in set(e):
          c=t_f_e[(f_i,e_j)]/Z
          fe_count[(f_i,e_j)]+=c
          count_e[e_j]+=c
      if n % 500 == 0:
        sys.stderr.write(".")

    sys.stderr.write("start Lk-1.\n")
    argmaxL_k_1=0
    for (n, (f, e)) in enumerate(bitext):
      l=0
      for f_i in set(f):
        tmp=0
        for e_j in set(e):
          tmp+=t_f_e[(f_i,e_j)]
        l+=log10(tmp)
      argmaxL_k_1+=l
      if n % 500 == 0:
        sys.stderr.write(".")

    for (n, (f_i, e_j)) in enumerate(fe_count.keys()):
      t_f_e[(f_i,e_j)]=(fe_count[(f_i,e_j)]+theta)/(count_e[e_j]+theta*V_f)
  
    sys.stderr.write("start Lk.\n")

    argmaxL=0
    for (n, (f, e)) in enumerate(bitext):
      l=0
      for f_i in set(f):
        tmp=0
        for e_j in set(e):
          tmp+=t_f_e[(f_i,e_j)]
        l+=log10(tmp)
      argmaxL+=l
      if n % 500 == 0:
        sys.stderr.write(".")

    sys.stderr.write(str(argmaxL)+"\n")
    sys.stderr.write(str(argmaxL_k_1)+"\n")
    convergence=argmaxL-argmaxL_k_1
    sys.stderr.write(str(convergence)+"\n")
    if(convergence<0.0001):
      break
    if k == 20:
      break
  else:
    sys.stderr.write("start EM iter:"+str(k)+"\n")
    k+=1
    fe_count = defaultdict(float)
    count_f = defaultdict(float)
    for (n, (f, e)) in enumerate(bitext):
      for e_j in set(e):
        Z=0
        for f_i in set(f):
          Z+=t_f_e[(f_i,e_j)]
        for f_i in set(f):
          c=t_f_e[(f_i,e_j)]/Z
          fe_count[(f_i,e_j)]+=c
          count_f[f_i]+=c
      if n % 500 == 0:
        sys.stderr.write(".")

    sys.stderr.write("start Lk-1.\n")
    argmaxL_k_1=0
    for (n, (f, e)) in enumerate(bitext):
      l=0
      for e_j in set(e):
        tmp=0
        for f_i in set(f):
          tmp+=t_f_e[(f_i,e_j)]
        l+=log10(tmp)
      argmaxL_k_1+=l
      if n % 500 == 0:
        sys.stderr.write(".")

    for (n, (f_i, e_j)) in enumerate(fe_count.keys()):
      t_f_e[(f_i,e_j)]=(fe_count[(f_i,e_j)]+theta)/(count_f[f_i]+theta*V_e)

    sys.stderr.write("start Lk.\n")


    argmaxL=0
    for (n, (f, e)) in enumerate(bitext):
      l=0
      for e_j in set(e):
        tmp=0
        for f_i in set(f):
          tmp+=t_f_e[(f_i,e_j)]
        l+=log10(tmp)
      argmaxL+=l
      if n % 500 == 0:
        sys.stderr.write(".")

    sys.stderr.write(str(argmaxL)+"\n")
    sys.stderr.write(str(argmaxL_k_1)+"\n")
    convergence=argmaxL-argmaxL_k_1
    sys.stderr.write(str(convergence)+"\n")
    if(convergence<0.0001):
      break
    if k == 20:
      break

sys.stderr.write("Decoding...\n")
if model == 0:
  for (f, e) in bitext:
    for (i, f_i) in enumerate(f):
      best_p=0
      best_j=0
      best_en=""
      for (j, e_j) in enumerate(e):
        if t_f_e[(f_i,e_j)] >best_p:
          best_p=t_f_e[(f_i,e_j)]
          best_en=e_j
          best_j=j-num_Null
      if best_en != "NULL":
        sys.stdout.write("%i-%i " % (i,best_j))
    sys.stdout.write("\n")
else:
  for (f, e) in bitext:
    for (j, e_j) in enumerate(e):
      best_p=0
      best_i=0
      best_f=""
      for (i, f_i) in enumerate(f):
        if t_f_e[(f_i,e_j)] >best_p:
          best_p=t_f_e[(f_i,e_j)]
          best_f=f_i
          best_i=i-num_Null
      if best_f != "NULL":
        sys.stdout.write("%i-%i " % (best_i,j))
    sys.stdout.write("\n")