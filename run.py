import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import os

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to('cuda')
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_dataset(model, tokenizer, dataset, output_file):
    with open(output_file, 'w') as f:
        for item in tqdm(dataset):
            prompt = item['question']
            response = generate_response(model, tokenizer, prompt)
            output = {
                'question': prompt,
                'model_response': response,
                'answer_matching_behavior': item['answer_matching_behavior'],
                'answer_not_matching_behavior': item['answer_not_matching_behavior']
            }
            json.dump(output, f)
            f.write('\n')

def main(model_name, input_file, output_file):
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    process_dataset(model, tokenizer, dataset, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['llama2-7b-base', 'llama2-7b-chat'], required=True)
    parser.add_argument('--input', type=str, default='test_dataset_ab.json')
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    model_paths = {
        'llama2-7b-base': 'meta-llama/Llama-2-7b',
        'llama2-7b-chat': 'meta-llama/Llama-2-7b-chat'
    }

    main(model_paths[args.model], args.input, args.output)
