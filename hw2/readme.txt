********************************************************************************************
										
			HW 2	: Phrasal Chunking											     	
			Group	: Amaz4				     	     
			Members	: Rao Fu, Yongyi Wu 			      
				  	  Song Yao, Zhelun Wu 				
										
********************************************************************************************
Run
___

python bigram_avgperc_chunker.py -n 13

-------------------------------------------------------------------------------------------

We had already generated the models from iteration #1~#18 in /models, and the corresponding outputs can be found in /outputs. 

The scores of each iteration using the submitted model are listed below: (Actually we tried other methods - trigram model specifically, but it seems that the simpler the model is, the better it performs) 

		    Ref250 Score|  Full Test Score  		 
iteration#		1:   90.15  |  90.02		
			2:   91.70  |  91.33
			3:   92.43  |  92.34
			4:   92.81  |  92.52
			5:   92.77  |  92.70
			6:   92.59  |  92.78
			7:   93.31  |  93.07
			8:   93.24  |  93.46
			9:   93.41  |  93.34
			10:  93.53  |  93.42
			11:  93.41  |  93.51
			12:  93.72  |  93.56
		*	13:  93.82  |  93.67	*  (Same as that mentioned in Collins 2002)
			14:  93.69  |  93.61
			15:  93.90  |  93.62
			16:  93.87  |  93.60
			17:  93.89  |  93.66
		*	18:  93.94  |  93.67    *

============================== Iteration# = 13 Score Details ==============================
processed 250 sentences with 5460 tokens and 2997 phrases; found phrases: 3006; correct phrases: 2816
	ADJP: precision:  75.93%; recall:  77.36%; F1:  76.64; found:     54; correct:     53
	ADVP: precision:  75.53%; recall:  73.96%; F1:  74.74; found:     94; correct:     96
	CONJP: precision:   0.00%; recall:   0.00%; F1:   0.00; found:      0; correct:      2
	NP: precision:  94.27%; recall:  94.57%; F1:  94.42; found:   1588; correct:   1583
	P: precision:  96.40%; recall:  98.40%; F1:  97.39; found:    639; correct:    626
	PRT: precision:  63.64%; recall:  70.00%; F1:  66.67; found:     11; correct:     10
	SBAR: precision:  81.63%; recall:  80.00%; F1:  80.81; found:     49; correct:     50
	VP: precision:  95.27%; recall:  94.28%; F1:  94.77; found:    571; correct:    577
accuracy:  95.90%; precision:  93.68%; recall:  93.96%; F1:  93.82
Score: 93.82
============================================================================================


========================= Section 1. Algorithm Introduction ================================

According to Collins 2002 and the given pseudo-code, the key update function, or learning function is:

	w_s = w_s + \Phi_s(w_[i:n_j], t_[1:n_j]) - \Phi_s(w_[i:n_j], z_[1:n_j])

\Phi_s is a feature vector, and for each feature inside we represented as:
				
	feature = [ feature_id , Tag]   
		eg. [ FEAT U15:NN/IN , I-NP] or [FEAT U06:the/pound, B-NP]

For each feature, if it is seen c_1 times with correctly labeled tag and c_2 times with wrong output tag. We let the feature = feature + c_1 - c_2, which in other words, if we see the correct tag, add one to the feature entry else minus one:
	
	feat_vec[(feat,es_tag)] -= 1
    	feat_vec[(feat,st_tag)] += 1

And also we applied a local parameter averaging methods mentioned in

http://gul.gu.se/public/pp/public_courses/course38351/published/1360057354030/resourceId/19456476/content/9adb1f1e-52e4-48b4-8001-ada93be18089/9adb1f1e-52e4-48b4-8001-ada93be18089.html (At the bottom: The Averaged Perceptron)

which improves a little bit. This part will be discussed in later section.

============================== Section 2. Language Model ===================================

Hidden Markov Model & Perceptron Algorithms
___________________________________________

What can I say? Awesome. And time-consuming.

The Averaged Perceptron
_______________________

"It's a bit impractical to keep all versions of the weight vector in memory, and then taking the average at the end. Instead, we can build the average vector incrementally, updating it simultaneously while we're building the normal weight vector."

Therefore instead of increasing or decreasing by one every time, we add or minus a variable STEP:
			
			STEP = (N*T - count) / (N*T) 	

N is the number of iterations, T is the number of samples(sentences), count is from 0 to N*T, added every time when we finished update for one sample.]

I think in page 34, we have averaged_feature_vector list = feature_vector / N*T, and locally we can do this by adding 1 / N*T every time.

However, the given formula is more intuitively reasonable: In earlier learning phase, our model is not 'smart' and often make mistakes, therefore we should penalize or reward more on the weights. Later in as the 'count' increased, the model gathered more input and the weights are expanded and relatively accurate than before. Then it's better that we adjust the weights using a smaller step.

============================== Section 3. Files Structure ==================================

Files
_____
	
readme.txt			: ReadMe File

bigram_avgperc_chunker.py 	: Main Code
	
models/ 			: Empty. Models are too huge to submit

outputs/			: Folder that contains outputs from iteration#13 and #18

