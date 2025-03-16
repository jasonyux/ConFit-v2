from openai import OpenAI
# Get OpenAI API key at https://platform.openai.com/account/api-keys
import argparse
import json
import os
import random
import pandas as pd
from tqdm import tqdm
import re


parser = argparse.ArgumentParser()

parser.add_argument("--index", type=str, required=True, help="index of split file")

args = parser.parse_args()



job_csv = pd.read_csv("dataset/recruiting_data_new/all_jd_full_from_text_recruiting_data.csv")
job_dict=dict(zip(job_csv["jd_no"],job_csv["text_job"]))

resume_csv = pd.read_csv("dataset/recruiting_data_new/all_resume_full_desensitized_recruiting_data.csv")
resume_dict=dict(zip(resume_csv["user_id"],resume_csv["text_resume"]))


with open("dataset/recruiting_data_new/templates.json","r",encoding="utf-8") as f:
    templates=json.load(f)

api_key="your key"

client = OpenAI(api_key=api_key)
  


result = {}
keys_to_convert=json.load(open(f"dataset/recruiting_data_new/gpt-4o-mini-job/keys_part_{args.index}.json","r"))

for i in tqdm(range(len(keys_to_convert)),desc="Process:{}".format(args.index)):
    template=random.choice(templates)  
    key=keys_to_convert[i]
    job = job_dict[key]
           
    prompt="\n".join(["Here is a template pair of matching resume and job:",
                                "[The start of the example job]",
                                template["job"],
                                "[The end of the example job]",
                                "[The start of the example resume]",
                                template["resume"],
                                "[The end of the example resume]"
                               "You are a helpful assistant. Following the above example pair of job and resume, construct an ideal resume for the target job shown below. You should strictly follow the format of the given pairs, make sure the resume you give perfectly matches the target job, and directly return your answer in plain texts.",
                                "[The start of the target job]",
                                job,
                                "[The end of the target job]"             
                ])
                
        
    chat_completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
            
    tokens=chat_completion.usage.completion_tokens
    # print(tokens)
            
    constructed=chat_completion.choices[0].message.content
    # print(constructed)
    text = re.sub(r"\[The start of .*?\]\n", "", constructed)
    text = re.sub(r"\[The start of .*?\]", "", text)
    text = re.sub(r"\n\[The end of .*?\]", "", text)
    text = re.sub(r"\n\[The end of .*?", "", text)
    text = re.sub(r"\[The end of .*?", "", text)
    
    result[key] = text
        

    
    # if (i // 200)  == 0:
    #     with open("dataset/recruiting_data_new/gpt-4o-mini-job/converted_{}.json".format(args.index),"w",encoding="utf-8") as f:
    #         json.dump(result,f,indent=4, ensure_ascii=False)
            
with open("dataset/recruiting_data_new/gpt-4o-mini-job/converted_{}.json".format(args.index),"w",encoding="utf-8") as f:
            json.dump(result,f,indent=4, ensure_ascii=False)
        
        