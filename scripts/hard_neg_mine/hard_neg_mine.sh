export PYTHONPATH=$(pwd)
export TRANSFORMERS_CACHE=./

CUDA_VISIBLE_DEVICES=0 python src/utils/hard_negative_mining.py \
--model_path model_checkpoints/confit_v2/epoch=7-step=19040.ckpt.fp32 \
--file_path hard_negatives \
--resume_data_path dataset/recruiting_data_new/all_resume_full_desensitized_recruiting_data_new.csv \
--job_data_path dataset/recruiting_data_new/all_jd_full_from_text_recruiting_data_new.csv \
--classification_data_path dataset/recruiting_data_new/train_labeled_data.jsonl \
--valid_classification_data_path dataset/recruiting_data_new/valid_classification_data.jsonl \
--rank_resume_data_path dataset/recruiting_data_new/valid_rank_resume.json \
--rank_job_data_path dataset/recruiting_data_new/valid_rank_job.json \
--dataset_type recruiting_data_new \
--query_prefix "" \
--batch_size 16 \
--seed 42 \
--lower_bound 0.04 \
--upper_bound 0.03 \
--num_resume_features 8 \
--num_job_features 9 \
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
--file_name hard_negtive.json
