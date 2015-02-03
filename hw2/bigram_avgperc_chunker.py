"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair. 

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""
from __future__ import division
import perc
import sys, optparse, os
from collections import defaultdict

def perc_train(train_data, tagset, n):
    feat_vec = defaultdict(int)
    feat_avg_vec = defaultdict(int)
    # insert your code here
    # please limit the number of iterations of training to n iterations
    default_tag = tagset[0]# tag any word with 'B-NP' in the beginning
    num_sentence = len(train_data)
    num_words = 0
    count = 0
    for iteration in range(n):
        sent_index = 0;
        for sentence in train_data:#sentence = (labeled_list, feat_list) for each sentence
            sent_index += 1
            print '{0}\r'.format("\rIteration: %d/%d. Sentence: %d/%d\t" %(iteration+1, n, sent_index, num_sentence)),
          
            (labeled_list, feat_list) = sentence
            num_words += len(labeled_list)

            #compute tags based on current weights
            estimated_tags = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, default_tag)
            #the target 'right' tag list
            standard_tags = [item.split()[2] for item in labeled_list]

            if estimated_tags != standard_tags:
                st_prev = es_prev = 'B_-1'
                index = 0
                #reference: http://gul.gu.se/public/pp/public_courses/course38351/published/1360057354030/resourceId/19456476/content/9adb1f1e-52e4-48b4-8001-ada93be18089/9adb1f1e-52e4-48b4-8001-ada93be18089.html
                step = (n*num_sentence - count)*1.0/(n*num_sentence)
                for (st_tag, es_tag) in zip(standard_tags,estimated_tags): 
                    (index, feats) = perc.feats_for_word(index, feat_list)

                    for feat in feats:
                        #deal with feat B: according to the given output example.
                        if feat == 'B':
                            if st_prev != es_prev or st_tag != es_tag:
                                feat_vec[('B:'+es_prev, es_tag)] -= 1
                                feat_vec[('B:'+st_prev, st_tag)] += 1
                                feat_avg_vec[('B:'+es_prev, es_tag)] -= step
                                feat_avg_vec[('B:'+st_prev, st_tag)] += step
                                es_prev = es_tag
                                st_prev = st_tag
                                
                        else:
                            if st_tag != es_tag:
                                feat_vec[(feat,es_tag)] -= 1
                                feat_vec[(feat,st_tag)] += 1
                                feat_avg_vec[(feat,es_tag)] -= step
                                feat_avg_vec[(feat,st_tag)] += step
            count += 1
        perc.perc_write_to_file(feat_avg_vec, 'models/n' + str(iteration) +'avg_params.model')        

    return feat_avg_vec

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-n", "--numiterations", dest="n", default=int(10), help="number of iterations of training")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."
    feat_vec = perc_train(train_data, tagset, int(opts.n))
    perc.perc_write_to_file(feat_vec, opts.modelfile)

