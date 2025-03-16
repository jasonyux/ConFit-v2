"""this script converts parsed resume/job objects into a single text string
"""

from src.schema.resume import Resume, Experience, Projects
from src.schema.job import Job
from src.schema.base import is_entity_empty, BaseEntity
from src.constants import EMPTY_DATA
from tqdm.auto import tqdm
import tiktoken
import pickle
import argparse
import pandas as pd
import numpy as np
import copy


tokenizer = tiktoken.get_encoding("cl100k_base")


def process_single_resume(resume: Resume):
    # default
    resume_string = resume.desensitized_str()
    return resume_string


def process_single_resume_for_confit_v1(resume: Resume):
    # default
    resume_string = resume.desensitized_str_for_confit_v1()
    return resume_string


def process_single_job(job: Job):
    # default
    job_string = job.desensitized_str()
    return job_string


def process_single_job_for_confit_v1(job: Job):
    # default
    job_string = job.desensitized_str_for_confit_v1()
    return job_string


def truncate_to_max_length(data, tokenizer, max_length=512):
    tokens = tokenizer.encode(data)[:max_length]
    return tokenizer.decode(tokens)


def process_single_resume_short(resume: Resume, tokenizer):
    # some additional rules on top of resume.desensitized_str()
    SECTION_SEP = '## '
    LIST_SEP = '\n-----\n'

    to_string = ''
    for k, v in resume.__dict__.items():
        if k in ['user_id', 'personal_info', 'metadata']:
            continue

        data_str = ''
        if is_entity_empty(v):
            continue
        elif k == "experiences":
            # custom logic
            len_limits = {
                'description': 200,
                'company_info.licensing_scope': 150,
                'company_info.description': 150,
            }
            for i, vv in enumerate(v):
                exp_cloned: Experience = copy.deepcopy(vv)
                exp_cloned.description = truncate_to_max_length(
                    exp_cloned.description,
                    tokenizer,
                    len_limits['description'] if i == 0 else len_limits['description'] // 2
                )
                exp_cloned.company_info.licensing_scope = truncate_to_max_length(
                    exp_cloned.company_info.licensing_scope,
                    tokenizer,
                    len_limits['company_info.licensing_scope'] if i == 0 else len_limits['description'] // 2
                )
                exp_cloned.company_info.description = truncate_to_max_length(
                    exp_cloned.company_info.description,
                    tokenizer,
                    len_limits['company_info.description'] if i == 0 else len_limits['description'] // 2
                )
                data_str += f'{str(exp_cloned)}{LIST_SEP}'
        elif k == "projects":
            # custom logic
            len_limits = {
                'description': 200
            }
            for i, vv in enumerate(v):
                proj_cloned: Projects = copy.deepcopy(vv)
                proj_cloned.description = truncate_to_max_length(
                    proj_cloned.description,
                    tokenizer,
                    len_limits['description'] if i == 0 else len_limits['description'] // 2
                )
                data_str += f'{str(proj_cloned)}{LIST_SEP}'
        elif isinstance(v, list):
            for vv in v:
                if isinstance(vv, BaseEntity):
                    data_str += f'{str(vv)}{LIST_SEP}'
                else:
                    data_str += f'- {str(vv)}\n'
        else:
            data_str += str(v)
        data_str = data_str.strip()
        to_string += f"{SECTION_SEP}{k}\n{data_str}\n\n"
    to_string = to_string.strip()
    return to_string


def process_single_resume_short_for_confit_v1(resume: Resume, tokenizer):
    # some additional rules on top of resume.desensitized_str_for_confit_v1()
    LIST_SEP = '\n-----\n'

    to_string_dict = {}
    for k, v in resume.__dict__.items():
        if k in ['user_id', 'personal_info', 'metadata']:
            continue

        data_str = ''
        if is_entity_empty(v):
            to_string_dict[k] = EMPTY_DATA
            continue
        elif k == "experiences":
            # custom logic
            len_limits = {
                'description': 200,
                'company_info.licensing_scope': 150,
                'company_info.description': 150,
            }
            for i, vv in enumerate(v):
                exp_cloned: Experience = copy.deepcopy(vv)
                exp_cloned.description = truncate_to_max_length(
                    exp_cloned.description,
                    tokenizer,
                    len_limits['description'] if i == 0 else len_limits['description'] // 2
                )
                exp_cloned.company_info.licensing_scope = truncate_to_max_length(
                    exp_cloned.company_info.licensing_scope,
                    tokenizer,
                    len_limits['company_info.licensing_scope'] if i == 0 else len_limits['description'] // 2
                )
                exp_cloned.company_info.description = truncate_to_max_length(
                    exp_cloned.company_info.description,
                    tokenizer,
                    len_limits['company_info.description'] if i == 0 else len_limits['description'] // 2
                )
                data_str += f'{str(exp_cloned)}{LIST_SEP}'
        elif k == "projects":
            # custom logic
            len_limits = {
                'description': 200
            }
            for i, vv in enumerate(v):
                proj_cloned: Projects = copy.deepcopy(vv)
                proj_cloned.description = truncate_to_max_length(
                    proj_cloned.description,
                    tokenizer,
                    len_limits['description'] if i == 0 else len_limits['description'] // 2
                )
                data_str += f'{str(proj_cloned)}{LIST_SEP}'
        elif isinstance(v, list):
            for vv in v:
                if isinstance(vv, BaseEntity):
                    data_str += f'{str(vv)}{LIST_SEP}'
                else:
                    data_str += f'- {str(vv)}\n'
        else:
            data_str += str(v)
        data_str = data_str.strip()
        to_string_dict[k] = data_str
    return to_string_dict


def configure_resume_formatter(name: str):
    print(f"Using formatter [{name}] for resume")
    if name == "default":
        return process_single_resume
    elif name == "default_confit_v1":
        return process_single_resume_for_confit_v1
    elif name == "short":
        return lambda r: process_single_resume_short(r, tokenizer)
    elif name == "short_confit_v1":
        return lambda r: process_single_resume_short_for_confit_v1(r, tokenizer)
    else:
        raise ValueError(f"Unknown formatter: {name} for resume")


def configure_job_formatter(name: str):
    print(f"Using formatter [{name}] for job")
    if name == "default":
        return process_single_job
    elif name == "default_confit_v1":
        return process_single_job_for_confit_v1
    else:
        raise ValueError(f"Unknown formatter: {name} for job")


def main(args: argparse.Namespace):
    with open(args.parsed_job_path, "rb") as f:
        job_parsed = pickle.load(f)
    with open(args.parsed_resume_path, "rb") as f:
        resume_parsed = pickle.load(f)
    print(f"Loaded {len(job_parsed)} jobs and {len(resume_parsed)} resumes")

    resume_formatter = configure_resume_formatter(args.resume_formatter)
    job_formatter = configure_job_formatter(args.job_formatter)

    ### process resume
    all_processed_resumes = []
    all_lengths = []
    for r in tqdm(resume_parsed, desc="Processing resumes"):
        text = resume_formatter(r)
        if isinstance(text, dict):
            all_processed_resumes.append({
                'user_id': r.user_id,
                **text,
            })
            curr_len = sum([len(tokenizer.encode(v)) for v in text.values()])
            all_lengths.append(curr_len)
        else:
            assert isinstance(text, str)
            all_processed_resumes.append({
                'user_id': r.user_id,
                'text': text,
            })
            all_lengths.append(len(tokenizer.encode(text)))
    
    all_processed_resumes_df = pd.DataFrame(all_processed_resumes)
    all_processed_resumes_df.to_csv(args.resume_save_path, index=False)
    print(f"Saved processed resumes to {args.resume_save_path}")
    print(np.mean(all_lengths), np.std(all_lengths))

    ### process jobs
    all_processed_jobs = []
    all_lengths = []
    for j in tqdm(job_parsed, desc="Processing jobs"):
        text = job_formatter(j)
        if isinstance(text, dict):
            all_processed_jobs.append({
                'jd_no': j.job_id,
                **text,
            })
            curr_len = sum([len(tokenizer.encode(v)) for v in text.values()])
            all_lengths.append(curr_len)
        else:
            assert isinstance(text, str)
            all_processed_jobs.append({
                'jd_no': j.job_id,
                'text': text,
            })
            all_lengths.append(len(tokenizer.encode(text)))

    all_processed_jobs_df = pd.DataFrame(all_processed_jobs)
    all_processed_jobs_df.to_csv(args.job_save_path, index=False)
    print(f"Saved processed jobs to {args.job_save_path}")
    print(np.mean(all_lengths), np.std(all_lengths))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parsed_job_path",
        type=str,
        default="dataset/recruiting_data_0729/intermediate/job_parsed.pkl"
    )
    parser.add_argument(
        "--parsed_resume_path",
        type=str,
        default="dataset/recruiting_data_0729/intermediate/resume_parsed.pkl"
    )
    parser.add_argument(
        "--job_save_path",
        type=str,
        default="dataset/recruiting_data_0729/processed/all_jd_full.csv"
    )
    parser.add_argument(
        "--resume_save_path",
        type=str,
        default="dataset/recruiting_data_0729/processed/all_resume_full.csv"
    )
    parser.add_argument(
        "--resume_formatter",
        type=str,
        default="default",
        choices=["default", "default_confit_v1", "short", "short_confit_v1"]
    )
    parser.add_argument(
        "--job_formatter",
        type=str,
        default="default",
        choices=["default", "default_confit_v1"]
    )
    args = parser.parse_args()

    print('received arguments:', args)

    main(args)