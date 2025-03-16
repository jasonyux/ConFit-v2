import pandas as pd
import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--output", type=str, required=True, help="output dir for merged file")
parser.add_argument("--concat", type=bool, default=False, help="whether concat hypothetical resume with origianl job")

args = parser.parse_args()

job_csv = pd.read_csv("dataset/recruiting_data_new/all_jd_full_from_text_recruiting_data.csv")
job_dict=dict(zip(job_csv["jd_no"],job_csv["text_job"]))

to_save={"jd_no":[],"text_job":[]}
for i in range(1,17):
        with open(f"{args.output}/converted_{i}.json","r",encoding="utf-8") as f:
           convert = json.load(f)
          

        for key in convert.keys():
           
            to_save["jd_no"].append(key)
            if args.concat:
               text = job_dict[key]+"\n[Example resume]\n"+convert[key]
            else:
               text = convert[key]
           
            
            to_save["text_job"].append(text)      


to_save_csv = pd.DataFrame(to_save)

if args.concat:
   to_save_csv.to_csv(f"{args.output}/all_job_converted_concat.csv",index=False)
else:
   to_save_csv.to_csv(f"{args.output}/all_job_converted.csv",index=False)