from tqdm.auto import tqdm
from pathlib import Path
import concurrent.futures
import glob
import json
import openai
import regex
import jieba
import argparse
import pandas as pd


FIELDS_TO_EXTRACT = [
    "Job Title",
    "Job Description/Responsibilities",
    "Job Location",
    "Job Position Type",  # Full-time, Part-time, Internship, etc.
    "Required Qualifications/Skills",
    "Preferred Qualifications/Skills",
    "Company Name",
    "Company Description",
    "Company Location",
]

FIELDS_TO_EXTRACT_CHINESE = [
    "职位名称",
    "工作描述/职责",
    "工作地点",
    "职位类型",  # Full-time, Part-time, Internship, etc.
    "必需的资质/技能",
    "优先的资质/技能",
    "公司名称",
    "公司描述",
    "公司地点",
]

PROMPT_TEMPLATE = """
The following is a job post found online. Your task is to extract the relevant information from this job post.
If some information is missing, answer with "None".

Job Post A:
-----
{jd}
------
Information you need to extract from Job Post A:
{formatted_fields}
[END]

Extracted Information from Job Post A:
""".strip()

PROMPT_TEMPLATE_CHINESE = """
以下是一份在网上找到的工作描述。你的任务是从这份工作描述中提取相关信息。
如果有信息缺失，请回答“无”。

工作描述 A: 
-----
{jd}
------
你需要从工作描述 A 中提取的信息:
- 职位名称
- 工作描述/职责
- 工作地点
- 职位类型
- 必需的资质/技能
- 优先的资质/技能
- 公司名称
- 公司描述
- 公司地点
[END]

从工作描述 A 中提取的信息:
""".strip()


def is_jd_chinese(jd: str):
    all_tokens = jieba.cut(jd)
    num_all_tokens = 0
    num_chinese_tokens = 0
    for token in all_tokens:
        num_all_tokens += 1
        if regex.match(r'\p{Han}+', token):
            num_chinese_tokens += 1
    if num_chinese_tokens > 0.1 * num_all_tokens:
        return True
    return False


def save_extracted_info(jid: str, data_dict: dict):
    save_path = f"dataset/recruiting_data_0514/raw/jobs_json_openai/{jid}.json"
    with open(save_path, "w", encoding="utf-8") as fwrite:
        json.dump(data_dict, fwrite, ensure_ascii=False, indent=4)
    return


def openai_generate_extract_info(jd_text: str, is_chinese: bool, model_name: str):
    if is_chinese:
        formatted_fields = '\n'.join([f'- {field}' for field in FIELDS_TO_EXTRACT_CHINESE])
        prompt = PROMPT_TEMPLATE_CHINESE.format(jd=jd_text, formatted_fields=formatted_fields)
        prefix = f'\n- {FIELDS_TO_EXTRACT_CHINESE[0]}:'
    else:
        formatted_fields = '\n'.join([f'- {field}' for field in FIELDS_TO_EXTRACT])
        prompt = PROMPT_TEMPLATE.format(jd=jd_text, formatted_fields=formatted_fields)
        prefix = f'\n- {FIELDS_TO_EXTRACT[0]}:'
    prompt += prefix

    gen_kwargs = {}
    client = openai.Client(base_url="https://api.openai.com/v1")

    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            n=1,
            timeout=60,  # added
            **gen_kwargs,
        )
        output = response.choices[0].message.content

        return {
            'output': output,
            'full_output': prefix + '' + output,
            'completion_tokens': response.usage.completion_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
        }
    except Exception as e:
        print('Error:', e)
        return {
            'output': 'ERROR',
            'full_output': 'ERROR',
            'completion_tokens': 0,
            'prompt_tokens': 0,
        }


def openai_extract_info(idx, jd: str, prompting_model: str, save_path: str):
    #### 1. prompt
    is_chinese = is_jd_chinese(jd)
    generated_info = openai_generate_extract_info(jd, is_chinese, prompting_model)

    #### 2. parse from text and extract
    gen_full_text = generated_info['full_output']
    all_gen_info_lines = gen_full_text.split('\n')

    field_names = FIELDS_TO_EXTRACT_CHINESE if is_chinese else FIELDS_TO_EXTRACT
    output_info = {f: [] for f in field_names}

    curr_field = field_names[0]
    for l in all_gen_info_lines:
        for f in field_names:
            if l.startswith(f'- {f}:'):
                curr_field = f
                break
        
        l = l.replace(f'- {curr_field}:', '').strip()
        if len(l) > 0:
            output_info[curr_field].append(l)
    
    for f in field_names:
        output_info[f] = '\n'.join(output_info[f])

    if is_chinese:
        output_info = {
            FIELDS_TO_EXTRACT[i]: output_info[f] for i, f in enumerate(FIELDS_TO_EXTRACT_CHINESE)
        }

    ####### 3. save
    save_file_path = f"{save_path}/{idx}.json"
    with open(save_file_path, "w", encoding="utf-8") as fwrite:
        json.dump(output_info, fwrite, ensure_ascii=False, indent=4)
    
    return {
        'jid': idx,
        'extracted_info': output_info,
        'completion_tokens': generated_info['completion_tokens'],
        'prompt_tokens': generated_info['prompt_tokens'],
    }


def estimate_current_cost(buffer_results):
    ### calculate cost
    total_completion_tokens = 0
    total_prompt_tokens = 0
    for res in buffer_results:
        total_completion_tokens += res['completion_tokens']
        total_prompt_tokens += res['prompt_tokens']
    print('Total completion tokens per data:', total_completion_tokens / len(buffer_results))
    print('Total prompt tokens per data:', total_prompt_tokens / len(buffer_results))
    return


def extract_info(args: argparse.Namespace, jid_to_job_dict: dict):
    num_parallel = 16
    model_name = args.model
    NUM_TO_EXTRACT = len(jid_to_job_dict)  # 5

    new_path = Path(args.source_folder).parent / 'jobs_json_openai'
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = []
        for jid, jd_text in jid_to_job_dict.items():
            future = executor.submit(
                openai_extract_info,
                jid,
                jd_text,
                model_name,
                new_path
            )
            futures.append(future)
            if len(futures) >= NUM_TO_EXTRACT:
                break  # used for debugging
        
        buffer_res = []
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            buffer_res.append(future.result())

            if (len(buffer_res) + 1) % 100 == 0:
                estimate_current_cost(buffer_res)
                buffer_res = []
    return


def prepare_data(args: argparse.Namespace):
    jid_to_include = None
    if args.filter_file != '':
        jid_to_include = pd.read_csv(args.filter_file)['job_id'].tolist()
        jid_to_include = set(jid_to_include)
        print('Including', len(jid_to_include), 'jobs')
    
    jid_to_job = {}
    for file in glob.glob(f"{args.source_folder}/*.json"):
        jid = file.split('/')[-1].split('.')[0]
        if jid_to_include is not None and jid not in jid_to_include:
            continue

        with open(file, 'r', encoding='utf-8') as f:
            jd = json.load(f)

        if 'text' not in jd:
            continue
        jid_to_job[str(jid)] = jd['text']
    
    
    done_jids = set()
    new_path = Path(args.source_folder).parent / 'jobs_json_openai'
    new_path.mkdir(parents=True, exist_ok=True)
    for jd_json in glob.glob(f'{new_path}/*.json'):
        with open(jd_json, 'r', encoding='utf-8') as f:
            jd = json.load(f)  # load it to check if it's valid
        jid = str(jd_json.split('/')[-1].split('.')[0])
        done_jids.add(jid)
    print('Found', len(done_jids), 'done jids')

    jid_to_job_todo = {jid: jd for jid, jd in jid_to_job.items() if jid not in done_jids}
    print('Total:', len(jid_to_job_todo))

    if args.max_data > 0:
        to_extract = min(args.max_data, len(jid_to_job_todo))
        jid_to_job_todo = {jid: jid_to_job_todo[jid] for jid in list(jid_to_job_todo.keys())[:to_extract]}
        print('To extract:', len(jid_to_job_todo))
    return jid_to_job_todo


if __name__ == "__main__":
    # example: python runners/data/extract_jd_openai.py --source_folder dataset/IIM_0624/data_jobs --filter_file dataset/IIM_0624/job_id_to_process.csv
    # this will extract 1200 jobs from the source_folder and save to dataset/IIM_0624/jobs_json_openai
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_data', type=int, default=-1)
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--filter_file', type=str, default='')
    parser.add_argument('--source_folder', type=str, default='dataset/recruiting_data_0514/raw/jobs')
    args = parser.parse_args()

    jid_to_job_todo = prepare_data(args)
    extract_info(args, jid_to_job_todo)