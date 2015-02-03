***********************************************************************************
										
			HW 1	: Word Segmentation											     	
			Group	: Amaz4				     	     
			Members	: Rao Fu, Yongyi Wu 			      
				  Song Yao, Zhelun Wu 				
										
***********************************************************************************

Run
___

python chart_based.py | python score-segments.py

OR 

python chart_based.py > output

==================== Section 1. Algorithm Introduction ============================

We use a 1-d vector: charts[] to maintain the segmentation. The basic idea is that charts[n] represents the best segmentation among the first n characters, so instead of generating candidates and putting them in the heap, we just look for previous positions and then choose the best segmentation. Besides we predefined a ‘max_word_len’, therefore we need to look for at most ‘max_word_len’ positions in charts[] before current position. 

eg. To determine word entry put in charts[n], we need to compare:
1) charts[n-1]+log_prob(text[n-1:n]),
2) charts[n-2]+log_prob(text[n-2:n]),
 … 
max_word_len) charts[n-max_word_len] + log_prob(text[n-max_word_len:n])

then we choose the max one, make an entry for it, and put it in charts[], after that continue on processing charts[n+1].

===================== Section 2. Language Model ===================================

Bigram Model Using Back-off Smoothing
_____________________________________

We set up bigram model using basic back-off smoothing, which is if we do not find the bigram word pair, just back off to unigram. (Actually we tried JM back off and set the lambda to 0.6, the best score we can get is around 90.10%)

Avoid Long Words
________________

And also, for unseen words, instead of assign them with same probability, we penalize on long words. So in avoid_long_words(), we use 10./(N * 10**len(word)) as the probability of unseen words. Besides, in avoid_long_words_II(), we use eight different parameters to control the penalty on word length, this part gives us quite a lot improvement and I beg you not to ask me why.

Learning Steps
______________

Intuitively, I think for the words that are chosen to be the best choice, they are also reasonably to be treated as valid words. And also if a combination happens multiple time, it probably to be a word, eg. People’s names. Therefore I added a learning step in the program: Every time we finished calculating for charts[i], we put the word from charts[i] in the dictionary. There are three cases:

1) word already in dictionary, and count > 1: then the word is in original input dictionary or have been seen for couple of times, add 1 to the count.

2) word already in dictionary, and count < 1: then the word is seen before but not exist in original dictionary, let count = count*1.1 + delta

3) word not exist in dictionary: assign dict[word] = delta.

With this learning step, our result increased a bit like 2% to 3%.
And I think it is reasonable since the program can always get score of above 80%(considering precision and recall), and we can believe in its choice.

Other Tricks
____________

Other tricks like string operations including:
1) Separate the text by comma ’，’ and Chinese comma ‘、’
2) Remove the spaces within numbers
3) Remove the spaces near ‘·’
*) We are going to use regular expression to split the text into smallest fractions, but it is not successful.

==================== Section 3. Files & Contributors ==============================

Files
_____
	
readme.txt
chart_based.py : It is the main file written by Yongyi Wu 
	
default_reviesed_newer_6819.py : It is prototype file provided by Yao Song

proto_raof.py : It is prototype file provided by Rao Fu 

Zhelun Wu worked on splitting text according to numbers and symbols.

========================== Section 4. chart_base.py ===============================

Global Varibles
_______________

[debug_model] : 
	True - Print results and messages about entries and charts[] for debugging.
	False - Print segmentation results.

[learnable]: Whether apply learning steps mentioned in Section 2.

[print_learning_dict]: Design for debug, whether print the key value pair when dictionary updated

[delta]: Learning parameter

[JM_back_off]: Whether apply Jelinek-Mercer smoothing
lam: lambda for Jelinek-Mercer smoothing, P_JM(w_i | w_{i-1}) = /lambda*P_ML(w_i| w_{i-1}) + ( 1 - /lambda)*P_JM(w_i)

[considering_last_character]: Not in use, I tried to generate a bigram dictionary using characters in count1w.txt, and also tried only consider the last character. Both not working very well. Actually I don’t think I understand the whole idea.

[word_max_len]: we assume the longest word has at most 8 characters.

[base]: base of logarithm probability.

=============================== Instructor’s Notes ================================

Run
---

python baseline.py | python score-segments.py

OR

python baseline.py > output
python score-segments.py -t output
rm output

Data
----

In the data directory, you are provided with counts collected from
training data which contained reference segmentations of Chinese
sentenes.

The format of the count_1w.txt and count_2w.txt is a tab separated key followed by a count:

__key__\t__count__

__key__ can be a single word as in count_1w or two words separated by a space as in count_2w.txt

