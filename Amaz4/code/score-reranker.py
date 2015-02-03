#!/usr/bin/env python
import optparse, sys, os
import bleu

optparser = optparse.OptionParser()
optparser.add_option("-r", "--reference", dest="reference", default=os.path.join("/usr/shared/CMPT/nlp-class/project/test/", "all.cn-en.en0"), help="English reference sentences")
optparser.add_option("-i", "--input", dest="input", default=os.path.join("/home/yongyiw/Documents/Github/final-project/Code", "output_1"), help="decoder output")
(opts,_) = optparser.parse_args()

# print opts.reference, opts.input
ref = [line.strip().split() for line in open(opts.reference)]
system = [line.strip().split() for line in open(opts.input)]

stats = [0 for i in xrange(10)]
for (r,s) in zip(ref, system):
  stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(s,r))]
print bleu.bleu(stats)
