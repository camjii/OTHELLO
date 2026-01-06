from typing import *
from openai import OpenAI
import os
from dotenv import load_dotenv
import re

# def check_if_same(text1):
#     '''ask query pipeline that checks 
#     if the given answer is part of a category'''



# def norm(text):
#     return text.lower()




# def hallucination_rate(pred, gold,length):
#     counter =0
#     pred_norm = norm(pred)
#     gold_norm = norm(gold)
#     if gold_norm == pred_norm or check_if_same(pred_norm) =='True':
#         counter +=0
#     else:
#         counter +=1
#     return counter/length #% of hallucinated q's
# 
# #def f1_score(pred, gold):
#     pred_tokens = set(pred.split())
#     gold_tokens = set(gold.split())
    
#     true_positives = len(pred_tokens.intersection(gold_tokens))
#     false_positives = len(pred_tokens - gold_tokens)
#     false_negatives = len(gold_tokens - pred_tokens)
    
#     if true_positives == 0:
#         return 0.0
    
#     precision = true_positives / (true_positives + false_positives)
#     recall = true_positives / (true_positives + false_negatives)
    
# #     f1 = 2 * (precision * recall) / (precision + recall)
# #     return f1

# pred = "McDonald's was founded by Richard McDonald and Maurice McDonald in 1940 in San Bernardino, California. They later partnered with Ray Kroc, who joined in 1954 and turned McDonald's into a global franchise."



# gold = "McDonald's was founded in 1940 by Richard McDonald and Maurice McDonald. They started the first McDonald's restaurant in San Bernardino, California. Later, Ray Kroc joined the company in 1955, expanded it nationwide through franchising, and eventually purchased the company, turning it into the global brand known today."
# # print("F1 Score:", f1_score(pred, gold))









load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()









def rephrase_using_question(answer, question): #API call
    prompt =  """
    Use the question to rephrase a single-word or single-sentence response into a full claim.
    Example:    Question: Who founded McDonald's and when?
                Response: Ray Kroc
                Rephrased Claim: Ray Kroc joined McDonald's in 1954.
    """
    
    
    response = client.chat.completions.create(
    model = 'gpt-4o-mini', 
    messages = [
        {"role": "system", "content": f"{prompt}"},
        {"role": "user", "content": f"Question: {question} Response: {answer} Rephrased Claim:"},
    ],
    temperature  = 0.3,
    max_tokens = 500
    )
   
    return response.choices[0].message.content.strip()




def split_into_claims(text:str) ->list: #API call
    
    prompt =  """
    "Parse the response into atomic claims, and verify the factuality of each, outputting a truthfulness score (0-100) for each claim."
    Example:  
                Response: "They later partnered with Ray Kroc, who joined in 1954 and turned McDonald's into a global franchise."
                Atomic Claims: ["Ray Crok joined McDonald's in 1954." Score: 95, "Ray Kroc expanded McDonald's into a global franchise. Score: 90"]
    """

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"Response: {text} Atomic Claims:"},
        ],
        max_completion_tokens=500,
        temperature =0.3
    )

    return response.choices[0].message.content.strip()



def extract_scores(claims_text: str) -> list:
    pattern = r'Score:\s*(\d{1,3})'
    scores = re.findall(pattern, claims_text)
    return [int(score) for score in scores]





# def check_claim_factuality(claim: str) -> bool: #API call
#     # API call to check factuality of a claim
#     return boolean  # Placeholder implementation



def main():

    question = "Who founded McDonald's and when?"

    # Given a test LLM response:
    answer = """McDonald's was founded by Richard McDonald and Maurice McDonald in 1940 in San Bernardino, California. They later partnered with Ray Kroc, who joined in 1954 and turned McDonald's into a global franchise."""
    
    # 1. Split the response into atomic claims.
    
    # If it's a single word or sentence, incorporate the question
    if len(answer.split('.')) == 1:
        answer = rephrase_using_question(answer, question)
    
     #2. For each claim, verify its factuality using reliable sources.
    
    print("Evaluating sentence:", answer)
    # for each sentence, ask judge LLM to identify all claims
    answer_claims = split_into_claims(answer)

    print("Extracted claims and scores:", answer_claims, extract_scores(answer_claims))






        
    # 3. Return a score from 0-100 based on the factual accuracy of the claims.
    if len(extract_scores(answer_claims)) == 0:
        return 0
    return sum(extract_scores(answer_claims))/len(extract_scores(answer_claims)) 

if __name__ == "__main__":
    score = main()
    print("Final factuality score:", score)

