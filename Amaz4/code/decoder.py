#!/usr/bin/env python
import optparse
import sys
import models_gz
from collections import namedtuple

def bitmap(sequence):
  """ Generate a coverage bitmap for a sequence of indexes """
  return reduce(lambda x,y: x|y, map(lambda i: long('1'+'0'*i,2), sequence), 0)

def bitmap2str(b, n, on='o', off='.'):
  """ Generate a length-n string representation of bitmap b """
  return '' if n==0 else (on if b&1==1 else off) + bitmap2str(b>>1, n-1, on, off)

def onbits(b):
  """ Count number of on bits in a bitmap """
  return 0 if b==0 else (1 if b&1==1 else 0) + onbits(b>>1)

def prefix1bits(b):
  """ Count number of bits encountered before first 0 """
  return 0 if b&1==0 else 1+prefix1bits(b>>1)

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="/usr/shared/CMPT/nlp-class/project/test/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="/usr/shared/CMPT/nlp-class/project/large/phrase-table/test-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="/usr/shared/CMPT/nlp-class/project/lm/en.gigaword.3g.filtered.dev_test.arpa.gz", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=5, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=100,type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]
# weight = [0.7763492,
# -0.5899425,
# 0.2942054,
# -0.4415968
# ]
weight = [21.3490854,
9.7672672,
12.3175896,
-7.363444]
models = models_gz
tm = models.TM(opts.tm, opts.k,weight [:4])
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0, [0.0, 0.0, 0.0 ,0.0])]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for (fucking_index, f) in enumerate(french):
  # print >> sys.stderr, '{0}\r'.format("\rSentence:%d/%d.\t" %(fucking_index, len(french))),


  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, bitmap")
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, 0)
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  distortion_limit=10
  for i, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune 
      firstopen= prefix1bits(h.bitmap)
      for j in xrange(firstopen,firstopen+1+distortion_limit):
        for k in xrange(j+1,len(f)+1):
          if f[j:k] in tm:
            new = bitmap(range(j,k))
            if h.bitmap & new ==0:
              v=new|h.bitmap
              for phrase in tm[f[j:k]]:
                logprob = h.logprob + phrase.logprob
                lm_state = h.lm_state
                for word in phrase.english.split():
                  (lm_state, word_logprob) = lm.score(lm_state, word)
                  logprob += word_logprob
                logprob += lm.end(lm_state) if k == len(f) else 0.0
                new_hypothesis = hypothesis(logprob, lm_state, h, phrase, v)
                index= onbits(v)
                key = (lm_state,v)
                if key not in stacks[index] or stacks[index][key].logprob < logprob: # second case is recombination
                  stacks[index][key] = new_hypothesis 

  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

  # def sum_features(h):
  #   return [0,0,0,0] if h.predecessor is None else [i+j for (i, j) in zip(  h.phrase.features , sum_features(h.predecessor) )]

  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)


  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))