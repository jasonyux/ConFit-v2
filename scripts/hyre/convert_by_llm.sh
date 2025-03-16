export PYTHONPATH=$(pwd) 


for i in {1..16}; do
    python src/utils/convert_by_llm_resume2job.py --index "$i" &  
done

wait  