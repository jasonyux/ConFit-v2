export PYTHONPATH=$(pwd)
export TRANSFORMERS_CACHE=./

CUDA_VISIBLE_DEVICES=0 python runners/trainer/train_inexit.py \
--save_path model_checkpoints/InEXIT \
--resume_data_path dataset/recruiting_data_v1/all_resume_full_desensitized_recruiting_data.csv \
--job_data_path dataset/recruiting_data_v1/all_jd_full_from_text_recruiting_data.csv \
--train_label_path dataset/recruiting_data_v1/train_labeled_data.jsonl \
--valid_label_path dataset/recruiting_data_v1/valid_classification_data.jsonl \
--classification_data_path dataset/recruiting_data_v1/test_classification_data.jsonl \
--rank_resume_data_path dataset/recruiting_data_v1/rank_resume.json \
--rank_job_data_path dataset/recruiting_data_v1/rank_job.json \
--dataset_type recruiting_data_v1 \
--train_batch_size 8 --val_batch_size 8 \
--pretrained_encoder jinaai/jina-embeddings-v2-base-zh \
--gradient_accumulation_steps 2 \
--weight_decay 1e-2 \
--log_group recruiting_data