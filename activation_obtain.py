import torch
from tqdm import tqdm
import argparse
import os
from datetime import datetime


def generation(tokenizer, model, prompt):
    # 获取激活值
    activations = {}  # 全局字典来存储激活值

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    # 注册钩子并存储句柄以便后续移除
    handles = []
    activation_list = []
    for i in range(model.config.num_hidden_layers):
        ffn_layer_name = f'model.model.layers.{i}.mlp.act_fn'  # 举例：Transformer的第12个隐藏层的FFN输入层
        handle = model.model.layers[i].mlp.act_fn.register_forward_hook(get_activation(ffn_layer_name))

        handles.append(handle)

    with torch.no_grad():
        tokens = tokenizer(prompt, padding=False, return_tensors="pt")
        input_ids = tokens.input_ids.to(model.device)

        # https://huggingface.co/docs/transformers/main_classes/text_generation
        output = model.generate(input_ids,
                                max_new_tokens=100,
                                return_dict_in_generate=True,
                                output_scores=True,
                                use_cache=True,
                                # output_logits=True,
                                # output_attentions=True,
                                # output_hidden_states=True,
                                # temperature=0.001,
                                # num_return_sequences=1,
                                do_sample=False, num_beams=1,

                                # top_k=10, 
                                # top_p=0.9,
                                # temperature=0.8,
                                # num_beams=1,
                                )

    prediction = tokenizer.batch_decode(output[0],
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0]
    # 移除钩子
    for handle in handles:
        handle.remove()

    # 获取所有激活值
    ffn_activations = [activations[name] for name in activations]

    return prediction, ffn_activations


def run_(r_file, w_file, prompt_template, get_answer, read_dataset, scorer, examplars_num, model_name,
         dataset, output_root_dir):
    from data_utils import write_model_out

    test_examples = read_dataset(file_path=r_file)

    is_correct = []
    progress = tqdm(test_examples[:examplars_num])

    for sample in progress:
        input_ = f"{sample['question']}"
        prompt = prompt_template.format(input_)
        prediction, activation_list = generation(tokenizer, model, prompt)

        sample["model_output"] = prediction.replace(prompt, '').split("\n\nQ")[0]
        sample['prediction'] = get_answer(sample["model_output"])
        sample['activation'] = [activation.reshape(-1).cpu().numpy().tolist() for activation in activation_list]

        is_correct.append(scorer(sample['prediction'], sample['answer']))
        sample['maj@k'] = is_correct[-1]
        sample['whether_in'] = 1 if sample['answer'] in sample['model_output'] else 0
        write_model_out(w_file, sample)
        score = sum(is_correct) / len(is_correct)
        progress.set_postfix(score=score)

    with open(os.path.join(output_root_dir, "log.txt"), "a", encoding="utf8") as f:
        f.write(model_name + "\n")
        f.write(w_file + "\n")
        f.write("acc: " + str(score) + "\n")
        f.write("*" * 20)
        f.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['gsm8k'],
                        help="please input the dataset")
    parser.add_argument('--prompt', type=str, required=True,
                        choices=['standard', 'cot'],
                        help='please input the type of prompt')
    parser.add_argument('--model', type=str, required=True, help="please specify the path of model")
    parser.add_argument('--num_test_sample', type=int, required=True,
                        help="please input the number of test samples (not exceed 100)")

    args = parser.parse_args()
    from data_utils import read_file_dataset
    from metrics import scorer
    from load_models import load_llm_model

    model, tokenizer = load_llm_model(args.model)
    if args.dataset == "gsm8k":
        r_file = "../dataset/MWP/GSM8K/random-100.jsonl"
        from prompt_GSM8K import GSM8K_PROMPT_cot, GSM8K_PROMPT_standard
        from data_utils import get_answer_mgsm_cot, get_answer_mgsm_standard

        if args.prompt == "cot":
            prompt_template = GSM8K_PROMPT_cot
            get_answer = get_answer_mgsm_cot
        elif args.prompt == "standard":
            prompt_template = GSM8K_PROMPT_standard
            get_answer = get_answer_mgsm_standard
    else:
        pass
    print(f"dataset={args.dataset}, prompt={args.prompt}")

    model_name = args.model.split("/")[-1].strip()
    num_test_sample = args.num_test_sample
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    read_dataset = read_file_dataset
    score = scorer

    output_root_dir = os.path.join("../output_activation", model_name)
    if not os.path.exists(output_root_dir):
        os.mkdir(output_root_dir)

    w_file = os.path.join(output_root_dir,
                          'activation_dataset_' + args.dataset + "_prompt_" + args.prompt + "_tested_" + str(
                              num_test_sample) + '_' + str(timestamp) + ".jsonl")
    run_(r_file, w_file, prompt_template, get_answer, read_dataset, scorer, args.num_test_sample, model_name,
         args.dataset, output_root_dir)
