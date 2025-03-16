from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, asdict
from src.preprocess.dataset import RJPairSimplifiedDataset, rj_pair_collate_fn
from src.config.dataset import DATASET_CONFIG
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from src.model.confit import (
    ConFitModel,
    ConFitModelArguments
)
import pandas as pd
import torch
import json
import numpy as np
import faiss
import os
from tqdm.auto import tqdm
import random
import scipy.stats as stats

os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclass
class HardNegativeSimArguments:
    model_path: str = field(
        default="model_checkpoints/dual_encoder_sft/aliyun_1e-9_decay",
        metadata={"help": "Path to the model we are evaluating and will dump results there."},
    )
    resume_data_path: str = field(
        default="dataset/AliTianChi/all_resume_w_updated_colnames.csv",
        metadata={"help": "Path to the resume data."},
    )
    job_data_path: str = field(
        default="dataset/AliTianChi/all_job_w_updated_colnames.csv",
        metadata={"help": "Path to the job data."},
    )
    classification_data_path: str = field(
        default="dataset/AliTianChi/train_classification_data.jsonl",
        metadata={"help": "Path to the train classification data."},
    )
    dataset_type: str = field(
        default="AliTianChi",
        metadata={"help": "Type of the dataset."},
    )
    query_prefix: str = field(
        default="",
        metadata={"help": "Query prefix to use."},
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for testing"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Seed for testing"},
    )
    lower_bound: float = field(
        default=0.7,
        metadata={"help": "Lower bound for hard negative mining."},
    )
    upper_bound: float = field(
        default=0.9,
        metadata={"help": "Upper bound for hard negative mining."},
    )
    file_path: str = field(
        default="",
        metadata={"help": "Path to the model we are evaluating and will dump results there."},
    )
    rank_resume_data_path: str = field(
        default="",
        metadata={"help": "Path to the rank resume data."},
    )
    rank_job_data_path: str = field(
        default="",
        metadata={"help": "Path to the rank job data."},
    )
    file_name: str = field(
        default="",
        metadata={"help": "File name for saving the results."},
    )

    def __post_init__(self):
        assert self.dataset_type in ["AliTianChi", "recruiting_data"], f"Invalid dataset type: {self.dataset_type}"
        assert self.dataset_type in self.resume_data_path, f"Dataset type {self.dataset_type} does not match resume data path {self.resume_data_path}"
        assert self.dataset_type in self.job_data_path, f"Dataset type {self.dataset_type} does not match job data path {self.job_data_path}"
        assert self.dataset_type in self.classification_data_path, f"Dataset type {self.dataset_type} does not match classification_data_path {self.classification_data_path}"
        return


def load_data(hard_negative_args: HardNegativeSimArguments):
    all_resume_data = pd.read_csv(hard_negative_args.resume_data_path)
    all_job_data = pd.read_csv(hard_negative_args.job_data_path)
    all_resume_data_dict = all_resume_data.to_dict("records")
    all_job_data_dict = all_job_data.to_dict("records")
    all_train_pairs = []
    seen_uid = set()
    seen_jdno = set()
    train_pairs = []
    
    classification_labels = pd.read_json(hard_negative_args.classification_data_path,lines=True)

    for i, row in classification_labels.iterrows():
        user_id = row["user_id"]
        jd_no = row["jd_no"]
        satisfied = row["satisfied"]
        train_pairs.append(
            {"user_id": str(user_id), "jd_no": str(jd_no), "satisfied": int(satisfied)}
        )
        if user_id in seen_uid and jd_no in seen_jdno:
            continue
        
        all_train_pairs.append(
            {"user_id": str(user_id), "jd_no": str(jd_no), "satisfied": int(satisfied)}
        )
        seen_uid.add(user_id)
        seen_jdno.add(jd_no)
            
    if hard_negative_args.rank_resume_data_path:
        ranking_resume_labels = json.load(open(hard_negative_args.rank_resume_data_path, encoding="utf-8"))
        for jd_no, data in ranking_resume_labels.items():
            user_ids = data["user_ids"]
            satisfieds = data["satisfied"]
            for user_id, satisfied in zip(user_ids, satisfieds):
                if user_id in seen_uid and jd_no in seen_jdno:
                    continue
                seen_uid.add(user_id)
                seen_jdno.add(jd_no)
    if hard_negative_args.rank_job_data_path:
        ranking_job_labels = json.load(open(hard_negative_args.rank_job_data_path, encoding="utf-8"))
        for user_id, data in ranking_job_labels.items():
            jd_nos = data["jd_nos"]
            satisfieds = data["satisfied"]
            for jd_no, satisfied in zip(jd_nos, satisfieds):
                if user_id in seen_uid and jd_no in seen_jdno:
                    continue
                seen_uid.add(user_id)
                seen_jdno.add(jd_no)

    print(f"Total number of pairs to compute: {len(all_train_pairs)}")
    return {
        "all_resume_data_dict": all_resume_data_dict,
        "all_job_data_dict": all_job_data_dict,
        "all_pairs": all_train_pairs,
    }, classification_labels


def generate_embeddings(model, model_args, hard_negative_args: HardNegativeSimArguments, all_data: dict):
    config = DATASET_CONFIG[hard_negative_args.dataset_type]
    max_seq_len_per_feature = config["max_seq_len_per_feature"]
    resume_taxon_token = config["resume_taxon_token"]
    job_taxon_token = config["job_taxon_token"]
    resume_key_names = config["resume_key_names"]
    job_key_names = config["job_key_names"]
    query_prefix = hard_negative_args.query_prefix
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_encoder)

    train_dataset = RJPairSimplifiedDataset(
        tokenizer,
        max_seq_length_per_key=max_seq_len_per_feature,
        resume_key_names=resume_key_names,
        job_key_names=job_key_names,
        tokenizer_args={"padding": "max_length", "return_tensors": "pt", "truncation": True},
        all_resume_dict=all_data["all_resume_data_dict"],
        all_job_dict=all_data["all_job_data_dict"],
        label_pairs=all_data["all_pairs"],
        resume_taxon_token=resume_taxon_token,
        job_taxon_token=job_taxon_token,
        query_prefix=query_prefix,
        encode_all = model_args.encode_all
    )

    dataloader = DataLoader(
        dataset=train_dataset,
        num_workers=4,
        batch_size=hard_negative_args
    .batch_size,
        shuffle=False,
        collate_fn=rj_pair_collate_fn,
    )

    rid_to_representation = {}
    jid_to_representation = {}

    model = model.to("cuda")
    model.eval()

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        base_idx = batch_idx * hard_negative_args.batch_size
        with torch.no_grad():
            output = model.predict_step(batch, batch_idx)

        batched_resume_representation = output.batched_resume_representation
        batched_job_representation = output.batched_job_representation

        for offset in range(batched_resume_representation.shape[0]):
            pair = all_data["all_pairs"][base_idx + offset]
            user_id = pair["user_id"]
            jd_no = pair["jd_no"]

            rid_to_representation[user_id] = batched_resume_representation[offset].cpu().numpy()
            jid_to_representation[jd_no] = batched_job_representation[offset].cpu().numpy()

    return rid_to_representation, jid_to_representation


def calculate_ranking_scores(rid_to_representation, jid_to_representation, all_pairs, lower = 0.04, upper = 0.03):
    rid_keys = list(rid_to_representation.keys())
    jid_keys = list(jid_to_representation.keys())

    rid_matrix = np.array([rid_to_representation[rid] for rid in rid_keys])
    jid_matrix = np.array([jid_to_representation[jid] for jid in jid_keys])

    faiss.normalize_L2(rid_matrix)
    faiss.normalize_L2(jid_matrix)

    index = faiss.IndexFlatIP(jid_matrix.shape[1])
    index.add(jid_matrix)

    
    scores, top_jids = index.search(rid_matrix, int(lower * len(jid_keys)))
    # scores, top_jids = index.search(rid_matrix, len(jid_keys))
    rid_to_top_jids = {}
    for i, rid in enumerate(rid_keys):
        rid_to_top_jids[rid] = [jid_keys[idx] for idx in top_jids[i][int(upper * len(jid_keys)):]]

    index = faiss.IndexFlatIP(rid_matrix.shape[1])
    index.add(rid_matrix)

    scores, top_rids = index.search(jid_matrix, int(lower * len(rid_keys)))
    # scores, top_rids = index.search(jid_matrix, len(rid_keys))
    
    jid_to_top_rids = {}

    for i, jid in enumerate(jid_keys):
        jid_to_top_rids[jid] = [rid_keys[idx] for idx in top_rids[i][int(upper * len(rid_keys)):]]
   
    # remove positive pairs 
    for _, label_data in all_pairs.iterrows():
        resume_id = str(label_data["user_id"])
        job_id = str(label_data["jd_no"])
        label = int(label_data["satisfied"])
        if label == 1:
            if resume_id in rid_to_top_jids.keys():
                if job_id in rid_to_top_jids[resume_id]:
                    rid_to_top_jids[resume_id].remove(job_id)
            if job_id in jid_to_top_rids.keys():
                if resume_id in jid_to_top_rids[job_id]:
                    jid_to_top_rids[job_id].remove(resume_id)
         
    return rid_to_top_jids, jid_to_top_rids


def save_ranking_results(r_to_j, j_to_r, file_path, file_name):
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, file_name), "w", encoding="utf-8") as f:
        json.dump({"r_to_j": r_to_j, "j_to_r": j_to_r}, f, indent=4)
        
        
def main(model_args: ConFitModelArguments, hard_negative_args: HardNegativeSimArguments):
    model = ConFitModel(model_args)
    state_dict = torch.load(hard_negative_args.model_path)["state_dict"]
    model.load_state_dict(state_dict)
    model = model.float()

    train_data, train_pairs = load_data(hard_negative_args)
    (rid_to_representation, jid_to_representation) = generate_embeddings(model, model_args, hard_negative_args, train_data)
    (rid_to_top_jids, jid_to_top_rids) = calculate_ranking_scores(rid_to_representation, jid_to_representation, train_pairs, hard_negative_args.lower_bound, hard_negative_args.upper_bound)
    save_ranking_results(rid_to_top_jids, jid_to_top_rids, hard_negative_args.file_path, hard_negative_args.file_name)
        
    
    
if __name__ == "__main__":
    parser = HfArgumentParser(dataclass_types=(HardNegativeSimArguments,ConFitModelArguments),)
    hard_negative_args, model_args  = parser.parse_args_into_dataclasses()
    print('received model args:')
    print(json.dumps(asdict(model_args), indent=2, sort_keys=True))
    print('received hard negative args:')
    print(json.dumps(asdict(hard_negative_args), indent=2, sort_keys=True))
    
    
    main(model_args, hard_negative_args)
    