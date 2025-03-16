import os
import random
import pandas as pd
from tqdm import tqdm
import re
import json

def split_list(input,output_dir, num_splits=16):
    num_splits = 16
    chunk_size = len(input) // num_splits + (1 if len(input) % num_splits != 0 else 0)


    
    os.makedirs(output_dir, exist_ok=True)

    # split to a number of json files for parallel convert
    for i in range(num_splits):
        chunk = input[i * chunk_size : (i + 1) * chunk_size] 
        output_file = os.path.join(output_dir, f"keys_part_{i+1}.json")  
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=4)  


job_csv = pd.read_csv("dataset/recruiting_data_new/all_jd_full_from_text_recruiting_data.csv")
job_dict=dict(zip(job_csv["jd_no"],job_csv["text_job"]))



## find keys to convert and split them, here we convert all, but you can define your keys to convert to reduce cost

keys_to_convert = list(job_csv["user_id"])
print("total number of jobs to convert:",len(keys_to_convert))
split_list(keys_to_convert,"dataset/recruiting_data_new/gpt-4o-mini-job",16)
print("keys split for parallel converting")