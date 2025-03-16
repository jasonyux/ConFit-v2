# ConFit

This repository contains an official implementation of ConFit V2, which is described in this paper:

**ConFit v2: Improving Resume-Job Matching using Hypothetical Resume Embedding and Runner-Up Hard-Negative Mining**<br>
*Xiao Yu\*, Ruize Xu\*, Chengyuan Xue\*, Jinzhong Zhang, Xu Ma, Zhou Yu*
\* Equal contribution

## Dependencies

1. install the required packages with `pip install -r requirements.txt`
2. make sure you have a `wandb` account and have logged in with `wandb login` (if not you need to remove the `wandb` related code in the scripts)
3. run `export PYTHONPATH=$(pwd)` `export TRANSFORMERS_CACHE=./`before running any scripts in this repository

## Data

<!-- All of our training data is placed under `dataset_pub/` folder. For privacy concerns, we replaced all the contents to placeholder such as `[[LONG_ENGLISH_TEXT]]`.
- If you would like to use our training script/model **with your own data**, you can simply follow the same format and replace the placeholder with your own data.
- If you would like to **have access to our data**, please contact us directly.

However, we note that **our dummy data should be compatible** with our provided scripts, since we only modified the *content*. So to test if you have set up everything correctly, you can directly try some of our examples in the [Training](#Training) section. -->

For greater control, we assume all data are parsed in the format of `.json`. The data format please refer to files in dataset folder. We merge all fields of texts of Recruiting dataset as recruiting_data_v2 for simplicity. For ConFIT v2, we recommend to use recruiting_data_v2 for HYRE and training.

## Hypothetical Resume Generation

We prompt LLMs (e.g. GPT-4o-mini) to generate a hypothetical resume given an input job. Here is the example of converting ```Recruiting_new``` dataset. You should adjust the script for others.

First run ```scripts/hyre/split_key.sh``` to split all job ids into subsets for later parallel processing. Then run ```scripts/hyre/convert_by_llm.sh``` to access openai APIs and prompt GPT-4o-mini to convert in parallel. Remember to fill in your own api key in ```src\utils\convert_by_llm.py```. Finally, run ```scripts/hyre/merge_json.sh``` to merge the converted results in a single csv file. Set ```concat=True``` if you want to concat the generated resume with the original job. 

To train\mine hard negatives with Hypothetical Resume, just change the ```job_data_path``` to the new merged file.

## Hard negatives mining
For hard negatives mining, run
```
bash scripts/hard_neg_mine/hard_neg_mine.sh
```
and set ```--model_path``` to the saved model's checkpoint trained without hard negatives.

## Training

### ConFit Training

The training scripts of ConFIT v2 with Jina-v2-base as the encoder are under ```scripts/confit_v2```. For example, to train ConFIT v2 on Recruiting data, you can run 
```
bash scripts/confit_v2/train_confitv2_recruit.sh
```

To use hard negatives during training, add an argument ```--hard_negative_path hard_negatives/hard_negatives.json ```, which corresponds to the mined hard negative file.

For AliYun dataset, you can similarly run 
```
bash scripts/confit_v2/train_confitv2_aliyun.sh
```

### Other Backbone

To use other backbones such as ```intfloat/multilingual-e5-base```, you should change the argument ```--pretrained_encoder <encoder name>```.  For ConFIT v2, you should also set the max length in ```src/config/proprietary_v2.py``` within input length limits of different models. For example, for e5 model, you should set 

```
_max_seq_len_per_resume_feature = {
    "text_resume": 512
}

_max_seq_len_per_job_feature = {
    "text_job": 512
    }
```


### Training Baselines

The training scripts of baselines are provided in ```scripts/baseline``` folder, including ```InEXIT```, ```MVCoN``` and ```ConFIT_v1```. Please refer to our paper for more details.

## Evaluation

There are many models and baselines we can evaluate. This not only includes evaluating the neural models we trained in the previous section, but also evaluating off-the-shelf embeddings and baselines such as BM25.


Evaluating XGBoost, BM25, and off-the-shelf embeddings:
- evaluate off-the-shelf embeddings. For example:
  ```bash
  python runners/tester/test_raw_embedding.py \
  --dset aliyun \
  --embedding_folder model_checkpoints/xlm-roberta/aliyun
  ```

- evaluate bm25. For example:
  ```bash
  python runners/tester/test_bm25.py --dset aliyun
  ```

Evaluating neural models we trained before

- evaluate ConFit v1:
  ```bash
  python runners/tester/test_confit.py \
  --model_path model_checkpoints/confit \
  --model_checkpoint_name epoch=2-step=975.ckpt \
  --resume_data_path dataset/recruiting_data/all_resume_w_flattened_text.csv \
  --job_data_path dataset/recruiting_data/all_jd.csv \
  --classification_validation_data_path dataset/recruiting_data/valid_classification_data.jsonl \
  --classification_data_path dataset/recruiting_data/test_classification_data.jsonl \
  --rank_resume_data_path dataset/recruiting_data/rank_resume.json \
  --rank_job_data_path dataset/recruiting_data/rank_job.json \
  --dataset_type recruiting_data \
  --wandb_run_id rcskbb25  # if you want to log the results to a wandb run
  ```
- evaluate InEXIT on the recruiting_data dataset:
  ```bash
  python runners/tester/test_inexit.py \
  --model_path model_checkpoints/InEXIT/recruiting_data_bce \
  --model_checkpoint epoch\=1-step\=1644.ckpt \
  --resume_data_path dataset/recruiting_data/all_resume_w_flattened_text.csv \
  --job_data_path dataset/recruiting_data/all_jd.csv \
  --classification_validation_data_path dataset/recruiting_data/valid_classification_data.jsonl \
  --classification_data_path dataset/recruiting_data/test_classification_data.jsonl \
  --rank_resume_data_path dataset/recruiting_data/rank_resume.json \
  --rank_job_data_path dataset/recruiting_data/rank_job.json \
  --dataset_type recruiting_data \
  --wandb_run_id uytnmv1q  # if you want to log the results to a wandb run
  ```
- evaluate MV-CoN on the recruiting_data dataset:
  ```bash
  python runners/tester/test_mvcon.py \
  --model_path model_checkpoints/MV-CoN/recruiting_data \
  --model_checkpoint epoch=7-step=22112.ckpt \
  --resume_data_path dataset/recruiting_data/all_resume_w_flattened_text.csv \
  --job_data_path dataset/recruiting_data/all_jd.csv \
  --classification_validation_data_path dataset/recruiting_data/valid_classification_data.jsonl \
  --classification_data_path dataset/recruiting_data/test_classification_data.jsonl \
  --rank_resume_data_path dataset/recruiting_data/rank_resume.json \
  --rank_job_data_path dataset/recruiting_data/rank_job.json \
  --dataset_type recruiting_data
  ```
## Citation
To cite our work, please use the BibTex below:
```
@misc{yu2025confitv2improvingresumejob,
      title={ConFit v2: Improving Resume-Job Matching using Hypothetical Resume Embedding and Runner-Up Hard-Negative Mining}, 
      author={Xiao Yu and Ruize Xu and Chengyuan Xue and Jinzhong Zhang and Xu Ma and Zhou Yu},
      year={2025},
      eprint={2502.12361},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12361}, 
}

```
