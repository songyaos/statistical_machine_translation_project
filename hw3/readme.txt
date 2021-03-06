********************************************************************************************
										
			HW 3	: Word Alignment											     	
			Group	: Amaz4				     	     
			Members	: Rao Fu, Yongyi Wu 			      
				  Song Yao, Zhelun Wu
				  An Yu 				
										
********************************************************************************************
Run
___

python AboveBaseline.py -p europarl -f de -n 10000 > output_iter20_0.0005_Null_3.a

python AboveBaseline.py -p europarl -f de -n 10000 > output_iter20_0.0005_Null_3_reverseST.a

head -1000 output_iter20_0.0005_Null_3.a > upload_iter20_0.0005_Null_3.a

head -1000 output_iter20_0.0005_Null_3_reverseST.a > upload_iter20_0.0005_Null_3_reverseST.a

python intersection.py > intersection.a
-------------------------------------------------------------------------------------------


We first find the alignment which is given English and then find the alignment which is given the German. In the AboveBaseline.py, there is a parameter called ‘model’. When model equals 0 means we align using Pr(g|e) and when model equals 1 means we align using Pr(e|g). At last, we remain the intersection of these two alignment sets as the final result. The ARE is 0.363 so far.

========================= Section 1. Methodology ================================

The main method used in this word alignment is EM algorithm. For getting a good result, we smoothed the probabilities using following formula:
	
	t(t|s)= (C(t,s)+n)/(c(s)+n*|V|)

In our code, we set n=0.0005 and |V| is the vocabulary size of target language.

Since each word in the target sentence can be generated by at most one word in the source sentence, we address the lack of sufficient alignments of target words to the null source word by adding extra null words to each source sentence. In our code, we added 3 null words for each source sentence. When decoding, the actual alignment index is the index minus the number of null words because we insert the null words at the beginning of the sentence.





