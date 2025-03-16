export PYTHONPATH=$(pwd)
export TRANSFORMERS_CACHE=./

CUDA_VISIBLE_DEVICES=0 python runners/trainer/train_confit.py \
--save_path model_checkpoints/confit_v2 \
--resume_data_path dataset/recruiting_data_v2/all_resume_full_desensitized_recruiting_data.csv \
--job_data_path dataset/recruiting_data_v2/all_jd_full_from_text_recruiting_data.csv \
--train_label_path dataset/recruiting_data_v2/train_labeled_data.jsonl \
--num_resume_features 1 \
--num_job_features 1 \
--valid_label_path dataset/recruiting_data_v2/valid_classification_data.jsonl \
--classification_data_path dataset/recruiting_data_v2/test_classification_data.jsonl \
--rank_resume_data_path dataset/recruiting_data_v2/rank_resume.json \
--rank_job_data_path dataset/recruiting_data_v2/rank_job.json \
--dataset_type recruiting_data_v2 \
--train_batch_size 4 \
--num_hard_negatives 2 \
--val_batch_size 4 \
--gradient_accumulation_steps 2 \
--weight_decay 1e-2 \
--num_suit_encoder_layers 0 \
--num_encoder_layers 0 \
--encode_all True \
--embedding_method mean_pool \
--finegrained_loss noop \
--pretrained_encoder jinaai/jina-embeddings-v2-base-zh \
--do_normalize true \
--temperature 0.05 \
--do_both_rj_hard_neg true \
--precision bf16  \
--strategy deepspeed_stage_2 \
--max_epochs 10