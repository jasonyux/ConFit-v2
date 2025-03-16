import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataclasses import dataclass, field
import os
import pickle
import json

@dataclass
class EmbeddingGenerationArguments:
    model_path: str = field(
        default="model_checkpoints/dual_encoder_sft/aliyun_1e-9_decay",
        metadata={"help": "Path to the model we are using to generate embeddings."},
    )
    resume_data_path: str = field(
        default="dataset/AliTianChi/all_resume_w_updated_colnames.csv",
        metadata={"help": "Path to the resume data."},
    )
    job_data_path: str = field(
        default="dataset/AliTianChi/all_job_w_updated_colnames.csv",
        metadata={"help": "Path to the job data."},
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for embedding generation"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Seed for embedding generation"},
    )

def load_data(args: EmbeddingGenerationArguments):
    resume_data = pd.read_csv(args.resume_data_path)
    job_data = pd.read_csv(args.job_data_path)
    return resume_data, job_data

def generate_embeddings(model, tokenizer, data, batch_size, device):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    embeddings = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

    return np.vstack(embeddings)

def cluster_embeddings(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

def find_hard_negatives(positive_pairs, resume_embeddings, job_embeddings, resume_labels, job_labels):
    hard_negatives = []
    for user_id, jd_no in positive_pairs:
        positive_resume_embedding = resume_embeddings[user_id]
        positive_job_embedding = job_embeddings[jd_no]
        
        same_cluster_negatives = [
            i for i, label in enumerate(resume_labels) if label != job_labels[jd_no]
        ]
        
        distances = np.linalg.norm(resume_embeddings[same_cluster_negatives] - positive_resume_embedding, axis=1)
        hard_negatives.append(same_cluster_negatives[np.argmin(distances)])

    return hard_negatives

def main(args: EmbeddingGenerationArguments):
    model = ...  # Load your model here
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    resume_data, job_data = load_data(args)
    
    resume_embeddings = generate_embeddings(model, tokenizer, resume_data, args.batch_size, "cuda")
    job_embeddings = generate_embeddings(model, tokenizer, job_data, args.batch_size, "cuda")
    
    resume_labels = cluster_embeddings(resume_embeddings)
    job_labels = cluster_embeddings(job_embeddings)
    
    positive_pairs = ...  # Load your positive pairs here
    hard_negatives = find_hard_negatives(positive_pairs, resume_embeddings, job_embeddings, resume_labels, job_labels)
    
    with open(os.path.join(args.model_path, "hard_negatives.pkl"), "wb") as f:
        pickle.dump(hard_negatives, f)

if __name__ == "__main__":
    args = EmbeddingGenerationArguments()
    main(args)
