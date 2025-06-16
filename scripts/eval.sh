model_name=""
model_type=""   # can be one of ['vllm', 'oai', 'hf']
model_path=""
max_length=""
data_path=""
output_dir="./results/"
data_parallel_size="1"
api_key=""  # OpenAI SDK
base_url=""
organization=""


python -m eval.eval --model_type "$model_type"\
    --model "$model_path"\
    --model_name "$model_name"\
    --max_length "$max_length"\
    --data_path "$data_path"\
    --output_dir "$output_dir"\
    --data_parallel_size "$data_parallel_size"\
    --api_key "$api_key"\
    --base_url "$base_url"\
    --organization "$organization"\