"""this script converts raw resume/job dict data into a single text string
"""
from tqdm.auto import tqdm
from typing import Callable, Any
from src.constants import EMPTY_DATA
from src.preprocess.utils import dict_to_text, dict_to_sectional_text
import jsonlines
import tiktoken
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from runners.data.recruiting_data.job_processor_recruiting_data import (
    generate_job_doc_dict,
    generate_job_text_dict,
)
from runners.data.recruiting_data.resume_processor_recruiting_data import TalentTrainingText 


tokenizer = tiktoken.get_encoding("cl100k_base")


def truncate_to_max_length(data, tokenizer, max_length=512):
    tokens = tokenizer.encode(data)[:max_length]
    return tokenizer.decode(tokens)


def configure_job_parser(name: str):
    print(f"Using formatter [{name}] for job")
    if name == "from_job_dict":
        return generate_job_doc_dict
    elif name == "from_job_text":
        return generate_job_text_dict
    else:
        raise ValueError(f"Unknown formatter: {name} for job")


def configure_resume_parser(name: str):
    print(f"Using formatter [{name}] for resume")
    if name == "all_fields":
        return "generate_talent_doc_dict"
    elif name == "desensitized":
        return "generate_talent_doc_dict_desensitized"
    else:
        raise ValueError(f"Unknown formatter: {name} for resume")


def configure_formatter(name: str):
    print(f"Using converting parsed dat to [{name}]")
    if name == "text":
        # use "\n" to join the outer keys, and for inner data use ; to separate
        return dict_to_text
    elif name == "dict":
        return dict_to_sectional_text
    else:
        raise ValueError(f"Unknown formatter: {name}")


def parse_job_from_json_file(job_json_path: str, parser_fn: Callable):
    with jsonlines.open(job_json_path) as reader:
        job_data = list(reader)
    ### process job
    all_job_parsed = []
    num_skipped = 0
    for record in job_data:
        job: dict = parser_fn(record["_source"]["snapShotJson"])
        jid = record['_id']

        if job is None or len(job) == 0:
            num_skipped += 1
            continue
        job['job_id'] = jid
        all_job_parsed.append(job)

    print(f"Skipped {num_skipped} jobs with {parser_fn.__name__}")
    print(f"Parsed {len(all_job_parsed)} jobs")
    return all_job_parsed


def parse_resume_from_json_file(resume_json_path: str, parser_fn: str):
    with jsonlines.open(resume_json_path) as reader:
        resume_data = list(reader)
    ### process resume
    all_resume_parsed = []
    num_skipped = 0
    for record in resume_data:
        resume_id = record['_id']

        ref_date = datetime.strptime(record["_source"]["earliestResumeCreatedDate"], '%Y-%m-%dT%H:%M:%SZ')
        t_ = TalentTrainingText(record["_source"]["resumeJson"], ref_date)
        if parser_fn == "generate_talent_doc_dict":
            resume = t_.generate_talent_doc_dict(truncate_seg=0)
        elif parser_fn == "generate_talent_doc_dict_desensitized":
            resume = t_.generate_talent_doc_dict_desensitized(truncate_seg=0)
        else:
            raise ValueError(f"Unknown parser_fn: {parser_fn}")
        
        if len(resume) == 0:
            num_skipped += 1
            continue
        resume['user_id'] = resume_id
        all_resume_parsed.append(resume)

    print(f"Skipped {num_skipped} resume with {parser_fn}")
    print(f"Parsed {len(all_resume_parsed)} jobs")
    return all_resume_parsed


def main_resume(args: argparse.Namespace):
    resume_parser = configure_resume_parser(args.resume_parser)
    # parse from raw. Also does some basic filtering
    resume_parsed = parse_resume_from_json_file(args.raw_resume_json_path, resume_parser)
    print(f"Converting {len(resume_parsed)} resume to text strings")
    
    formatter_fn = configure_formatter(args.resume_formatter)
    
    ### process resume
    all_processed_resumes = []
    all_lengths = []
    for r in tqdm(resume_parsed, desc="Processing resumes"):
        user_id = str(r.pop('user_id'))
        text = formatter_fn(r)
        if isinstance(text, dict):
            all_processed_resumes.append({
                'user_id': user_id,
                **text,
            })
            curr_len = sum([len(tokenizer.encode(v)) for v in text.values()])
            all_lengths.append(curr_len)
        else:
            assert isinstance(text, str)
            all_processed_resumes.append({
                'user_id': user_id,
                'resume_text': text,
            })
            all_lengths.append(len(tokenizer.encode(text)))

    all_processed_resumes_df = pd.DataFrame(
        all_processed_resumes
    )
    all_processed_resumes_df = all_processed_resumes_df.fillna(EMPTY_DATA)
    all_processed_resumes_df.to_csv(args.resume_save_path, index=False)
    print(f"Saved {len(all_processed_resumes_df)} processed resume to {args.resume_save_path}")
    print(np.mean(all_lengths), np.std(all_lengths))
    return


def main_job(args: argparse.Namespace):
    job_parser = configure_job_parser(args.job_parser)
    # parse from raw. Also does some basic filtering
    job_parsed = parse_job_from_json_file(args.raw_job_json_path, job_parser)
    print(f"Converting {len(job_parsed)} jobs to text strings")
    
    formatter_fn = configure_formatter(args.job_formatter)
    
    ### process jobs
    all_processed_jobs = []
    all_lengths = []
    for j in tqdm(job_parsed, desc="Processing jobs"):
        jid = str(j.pop('job_id'))
        text = formatter_fn(j)
        if isinstance(text, dict):
            all_processed_jobs.append({
                'jd_no': jid,
                **text,
            })
            curr_len = sum([len(tokenizer.encode(v)) for v in text.values()])
            all_lengths.append(curr_len)
        else:
            assert isinstance(text, str)
            all_processed_jobs.append({
                'jd_no': jid,
                'job_text': text,
            })
            all_lengths.append(len(tokenizer.encode(text)))

    all_processed_jobs_df = pd.DataFrame(
        all_processed_jobs
    )
    all_processed_jobs_df = all_processed_jobs_df.fillna(EMPTY_DATA)
    all_processed_jobs_df.to_csv(args.job_save_path, index=False)
    print(f"Saved {len(all_processed_jobs_df)} processed jobs to {args.job_save_path}")
    print(np.mean(all_lengths), np.std(all_lengths))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["resume", "job"]
    )
    parser.add_argument(
        "--raw_job_json_path",
        type=str,
        default="dataset/recruiting_data_0729/job_snapshot_240702_v2.json",
        help="Path to the raw job json file that comes from dumping elastic search"
    )
    parser.add_argument(
        "--job_save_path",
        type=str,
        default="dataset/recruiting_data_0729/processed_recruiting_data/all_jd_full_from_text_recruiting_data.csv"
    )
    parser.add_argument(
        "--job_parser",
        type=str,
        default="from_job_text",
        choices=["from_job_dict", "from_job_text"]
    )
    parser.add_argument(
        "--job_formatter",
        type=str,
        default="text",
        choices=["text", "dict"]  # dict is used by ConFit V1
    )
    parser.add_argument(
        "--raw_resume_json_path",
        type=str,
        default="dataset/recruiting_data_0729/resumes_240702_v2.json",
        help="Path to the raw resume json file that comes from dumping elastic search"
    )
    parser.add_argument(
        "--resume_save_path",
        type=str,
        default="dataset/recruiting_data_0729/processed_recruiting_data/all_resume_full_desensitized_recruiting_data.csv"
    )
    parser.add_argument(
        "--resume_parser",
        type=str,
        default="desensitized",
        choices=["all_fields", "desensitized"]
    )
    parser.add_argument(
        "--resume_formatter",
        type=str,
        default="text",
        choices=["text", "dict"]  # dict is used by ConFit V1
    )
    args = parser.parse_args()

    print('received arguments:', args)

    if args.mode == "resume":
        main_resume(args)
    elif args.mode == "job":
        main_job(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")