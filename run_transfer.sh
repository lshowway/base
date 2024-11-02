model_dir="/opt/models"
examplars_num=50
models=("gemma-2-2b" "gemma-2-9b" "gemma-2-27b" "Llama-2-13b-hf")
datasets=("aqua" "gsm8k" "svamp" "sports" "date" "bamboogle" "coinflip" "lastletters")
prompts=("standard" "cot" "aqua" "gsm8k" "svamp" "sports" "date" "bamboogle" "coinflip" "lastletters")

for model in "${models[@]}"; do
    model_path="$model_dir/$model"
    for dataset in "${datasets[@]}"; do
        for prompt in "${prompts[@]}"; do
            if [ "$dataset" != "$prompt" ]; then
                echo "Running run.py with model $model , dataset $dataset , prompt $prompt"
                python run.py --dataset "$dataset" --prompt "$prompt" --model $model_path --examplars $examplars_num
            fi
        done
    done
done 