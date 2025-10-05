#minicons for getting the model's surprisal scores 

from minicons import scorer
import pandas as pd
import numpy as np
import random

#MODELS
#ltg: https://doi.org/10.18653/v1/2023.findings-eacl.146
#baseline for second babylm and winner of first call
ltgbert_small=scorer.MaskedLMScorer('babylm/ltgbert-10m-2024', 'cpu', trust_remote_code=True)
ltgbert_strict=scorer.MaskedLMScorer('babylm/ltgbert-100m-2024', 'cpu', trust_remote_code=True)

#gpt-2:https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
#baseline for third call
gpt2_small=scorer.IncrementalLMScorer('BabyLM-community/babylm-baseline-10m-gpt2', 'cpu', trust_remote_code=True)
gpt2_strict=scorer.IncrementalLMScorer('BabyLM-community/babylm-baseline-100m-gpt2', 'cpu', trust_remote_code=True)

#roberba: https://doi.org/10.48550/arXiv.1907.11692
#baseline for first call
roberta_small=scorer.MaskedLMScorer('babylm/roberta-base-strict-small-2023', 'cpu', trust_remote_code=True)
roberta_strict=scorer.MaskedLMScorer('babylm/roberta-base-strict-2023', 'cpu', trust_remote_code=True)

#babyllama: https://doi.org/10.48550/arXiv.2308.02019
#baseline for second babylm
babyllama_small=scorer.IncrementalLMScorer('babylm/babyllama-10m-2024', 'cpu', trust_remote_code=True)
babyllama_strict=scorer.IncrementalLMScorer('babylm/babyllama-100m-2024', 'cpu', trust_remote_code=True)

#COMPUTATION
#We then capture the model's probability for each answer, 
#conditioned on the question, assuming that if the model shows more probability for dialogs with the follower's answers, 
#it's pragmatically plausible with respect to that maxim. 
def compute_probability(model, df, model_name):

  result_df=pd.DataFrame()

  questions=df["question"].to_list()
  following=df["follower"].to_list()
  violating=df["violator"].to_list()

  score_follower=[]
  score_violator=[]

  for i in range(0, len(questions), 150):
      q_chunk=questions[i:i+150]
      f_chunk=following[i:i+150]
      v_chunk=violating[i:i+150]

      f_scores=model.conditional_score(q_chunk, f_chunk, reduction=lambda x: x.mean(0).item())
      v_scores=model.conditional_score(q_chunk, v_chunk, reduction=lambda x: x.mean(0).item())

      score_follower.extend(f_scores)
      score_violator.extend(v_scores)

  accuracy=[f > v for f, v in zip(score_follower, score_violator)]

  result_df[f"{model_name} | P. Follower"]=score_follower
  result_df[f"{model_name} | P. violator"]=score_violator
  result_df[f"{model_name} | Accuracy (F. > V.)"]=accuracy

  return result_df

#EVAL DATASET
dataset=pd.read_csv("./eval_dataset.csv")

eval_ltgstrict=compute_probability(ltgbert_strict, dataset, "LTG BERT Strict")
eval_ltgsmall=compute_probability(ltgbert_small, dataset, "LTG BERT Small")
eval_gpt2small=compute_probability(gpt2_small, dataset, "GPT 2 Small")
eval_gpt2strict=compute_probability(gpt2_strict, dataset, "GPT 2 Strict")
eval_robertasmall=compute_probability(roberta_small, dataset, "Roberta_small")
eval_robertastrict=compute_probability(roberta_strict, dataset, "Roberta Strict")
eval_babyllamasmall=compute_probability(babyllama_small, dataset, "Baby Llama Small")
eval_babyllamastrict=compute_probability(babyllama_strict, dataset, "Baby Llama Strict")
small_models=pd.concat([dataset, eval_ltgsmall, eval_gpt2small, eval_robertasmall, eval_babyllamasmall], axis=1)
strict_models=pd.concat([dataset, eval_ltgstrict, eval_gpt2strict, eval_robertastrict, eval_babyllamastrict], axis=1)

small_models.to_csv("eval small models.csv", index=False)
strict_models.to_csv("eval strict models.csv", index=False)

smalls=["LTG BERT Small", "GPT 2 Small", "Roberta_small", "Baby Llama Small"]
stricts=["LTG BERT Strict", "GPT 2 Strict", "Roberta Strict", "Baby Llama Strict"]

overall_small=[]
for small in smalls:
  df=small_models.groupby("maxim")[f"{small} | Accuracy (F. > V.)"].mean()
  overall_small.append(df)

overall_strict=[]
for strict in stricts:
  df=strict_models.groupby("maxim")[f"{strict} | Accuracy (F. > V.)"].mean()
  overall_strict.append(df)

overall_small=pd.concat(overall_small, axis=1)
overall_strict=pd.concat(overall_strict, axis=1)

ordered_maxims=["Be informative", "Avoid redundant information",
    "Be truthful", "Be relevant", "Be polite"]

overall_small=overall_small.loc[ordered_maxims].T
overall_strict=overall_strict.loc[ordered_maxims].T

overall_small.to_csv("./overall evals for small models.csv", index=False)
overall_strict.to_csv("./overall evals for strict models.csv", index=False)

#LLM 
#we do the same for OLMO 1B
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
olmo1b = scorer.IncrementalLMScorer('allenai/OLMo-1B', 'cpu', trust_remote_code=True)
olmo1b_scores=compute_probability(olmo1b, dataset, "Olmo 1B")
llm_score=pd.concat([dataset, olmo1b_scores], axis=1)
llm_score=pd.read_csv("llm_score.csv")

overall_llm=llm_score.groupby("maxim")["Olmo 1B | Accuracy (F. > V.)"].mean()
ordered_maxims=["Be informative", "Avoid redundant information",
    "Be truthful", "Be relevant", "Be polite"]

overall_llm=overall_llm.loc[ordered_maxims].T
overall_llm.to_csv("overall llm.csv", index=False)

#Scores are now all saved in their respective files. 