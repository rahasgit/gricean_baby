# gricean_baby

This is the code for evaluating language models on the evaluation dataset available at \url{https://huggingface.co/datasets/rahaaskari/gricean_baby}, 
and for reproducing the results of our paper (link to be shared soon). 
The evaluation dataset consists of short conversations, with each having one question and two possible answers, one of which is less pragmatically plausible than the other.
The code computes the probability of the model for each answer, conditioned on the question. 
As a measure of accuracy, it then counts how many times the model assigns a higher probability to the correct answer. 
Please refer to our paper for more info. 
