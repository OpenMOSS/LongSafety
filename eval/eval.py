from .evaluator import Evaluator
from .model import OAIModel, HF_Model, VLLM

from argparse import ArgumentParser
import os
import torch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_type', type=str, default='hf')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_path', type=str, default='LutherXD/LongSafetyBench')
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.getcwd(), 'results'))
    parser.add_argument('--data_parallel_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=1024*32)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--organization', type=str, default=None)
    
    
    args = parser.parse_args()
    task = args.task
    data_path = args.data_path
    output_dir = args.output_dir
    
    # 获取可用 GPU 数量
    if args.model_type != 'oai':
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise RuntimeError("No GPUs are available.")
        
        gpus_per_model = gpu_count // args.data_parallel_size
        if gpus_per_model == 0:
            raise ValueError("Insufficient GPUs for the specified data_parallel_size.")
    else:
        gpus_per_model = args.data_parallel_size
    
    model_type = args.model_type.lower()
    assert model_type in ['hf', 'oai', 'vllm']
    models = []
    for i in range(args.data_parallel_size):
        assigned_gpus = list(range(i * gpus_per_model, (i + 1) * gpus_per_model))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpus))
        if model_type == 'vllm':
            model = VLLM(
                model=args.model,
                model_name=args.model_name,
                tensor_parallel_size=gpus_per_model,
                max_length=args.max_length
            )
        elif model_type == 'oai':
            model = OAIModel(
                model=args.model,
                model_name=args.model_name,
                api_key=args.api_key,
                base_url=args.base_url,
                organization=args.organization,
                max_length=args.max_length
            )
        elif model_type == 'hf':
            model = HF_Model(
                model=args.model,
                model_name=args.model_name,
                max_length=args.max_length
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        models.append(model)

    evaluator = Evaluator(
        model=models,
        output_dir=output_dir,
        data_path=data_path,
        load_local=False
    )

    evaluator.evaluate(task)