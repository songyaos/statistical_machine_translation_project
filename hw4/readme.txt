********************************************************************************************
										
			HW 4	: Translation Decoding											     	
			Group	: Amaz4				     	     
			Members	: Rao Fu, Yongyi Wu 			      
				  Song Yao, Zhelun Wu
				  An Yu 				
										
********************************************************************************************
Run
___

python decoder.py -s 1000 -k 20 > output
python score-decoder.py < output
-------------------------------------------------------------------------------------------


In this homework, we have two models for machine translation, one based on the translation of words, and another based on the translation of phrases as atomic units. It assigns score for each alignment with sum of log translation model and log language model.

========================= Section 1. Methodology ================================

We adopt beam search to find good translation according to assigned score. We expand a hypothesis when we pick one of the translation options and construct a new hypotheses.
In the data structure of hypotheses, we have attribute bitmap which is used for recording the coverage of translated words at current state. Every time we expand a hypotheses, we update the bitmap having or(|) operation with their father hypotheses bitmap. And also, every time we start translate the french words where the first zero occur in its father’s bitmap. 

Since we are using the stack decoding, the index of the stack which means how many words have been translated in french sentences and the key for each stack is the language model state with current bitmap.





