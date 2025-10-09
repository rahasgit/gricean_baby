# A Gricean Approach for Evaluation of Language Models

This is the code for evaluating language models on the evaluation dataset available at [Hugging Face](https://huggingface.co/datasets/rahaaskari/gricean_baby), 
and for reproducing the results of our [paper](https://arxiv.org/abs/2510.04764). 

The evaluation dataset consists of short conversations, with each having one question and two possible answers, one of which is less pragmatically plausible than the other.
The code computes the probability of the model for each answer, conditioned on the question. 
As a measure of accuracy, it then counts how many times the model assigns a higher probability to the correct answer. 

Please refer to our paper for more info. 
