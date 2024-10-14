import json
import os
import re
import pandas as pd


def read_file_dataset(file_path="../dataset/MWP/GSM8K/test.jsonl"):
    path = os.path.join(file_path)
    print("Loading " + path)
    with open(path, encoding='utf-8') as fh:
        # examples = [ for line in fh.readlines() if line]
        examples = [eval(line) for line in fh.readlines()]
    print(f"Total {len(examples)} examples")
    return examples


def read_file_mgsm(file_path="../dataset/MWP/GSM8K/test.jsonl"):
    path = os.path.join(file_path)
    print("Loading " + path)
    with open(path) as fh:
        # examples = [ for line in fh.readlines() if line]
        examples = [{"question": eval(line)["question"], "answer": eval(line)['answer'].split('#### ')[1].strip()} for
                    line in fh.readlines()]

    print(f"{len(examples)} examples")
    return examples


def read_file_svamp(file_path="../dataset/MWP/SVAMP/SVAMP.json"):
    json_data = {}
    examples = []
    with open(file_path, "r", encoding="utf8") as f:
        json_data = [eval(line) for line in f.readlines()]
    for item in json_data:
        body = item["Body"]
        if not body.endswith('.'):
            body += ". "
        item["question"] = body + ' ' + item["Question"]
        item["answer"] = str(item["Answer"])
        examples.append(item)
    print(f"{len(examples)} examples")
    return examples


def read_file_aqua(file_path="../dataset/MWP/AQUA/test.jsonl"):
    print("loading " + file_path)

    def replace_prefix(text):
        # 定义替换规则
        replacements = {
            'A)': '(a) ',
            'B)': '(b) ',
            'C)': '(c) ',
            'D)': '(d) ',
            'E)': '(e) '
        }

        # 使用正则表达式进行替换
        pattern = re.compile(r'^[ABCDE]\)')
        match = pattern.match(text)
        if match:
            prefix = match.group(0)
            text = text.replace(prefix, replacements[prefix])
        return text

    examples = []
    path = os.path.join(file_path)
    with open(path) as fh:
        for line in fh.readlines():
            q = json.loads(line)["question"].strip()
            options = ""
            temp = {}
            for option in json.loads(line)["options"]:
                options += replace_prefix(option)
                options += " "
            temp["question"] = q + "\n" + options
            temp["answer"] = json.loads(line)["correct"]
            examples.append(temp)
    print(f"{len(examples)} examples")
    return examples


def read_file_bamboogle(file_path="../dataset/CommonsenseReasoning/Bamboogle/bamboogle.csv"):
    path = os.path.join(file_path)
    data = pd.read_csv(path)
    data = data[["Question", "Answer", "Answer 2"]]
    examples = []
    for index, row in data.iterrows():
        dict_row = row.to_dict()
        dict_row["question"] = dict_row["Question"]
        dict_row["answer"] = dict_row["Answer"]
        examples.append(dict_row)
    print(f"{len(examples)} examples")
    return examples


def read_file_sports(file_path="../dataset/CommonsenseReasoning/BigBench_Sports_understanding/test.json"):
    json_data = {}
    with open(file_path, "r", encoding="utf8") as f:
        json_data = json.load(f)
    examples_1 = json_data["examples"]
    examples = []
    for item in examples_1:
        if item["target_scores"]["plausible"] == 1:
            item["correct"] = "plausible"
            item["answer"] = "yes"
        else:
            item["correct"] = "implausible"
            item["answer"] = "no"
        item["question"] = item["input"]
        examples.append(item)
    print(f"{len(examples)} examples")
    return examples


def read_file_strategyqa(file_path="../dataset/CommonsenseReasoning/StrategyQA/strategyqa.json"):
    json_data = {}
    with open(file_path, "r", encoding="utf8") as f:
        json_data = json.load(f)
    examples_1 = json_data["examples"]
    examples = []
    for item in examples_1:
        if item["target_scores"]["Yes"] == 1:
            item["correct"] = "plausible"
            item["answer"] = "yes"
        else:
            item["correct"] = "No"
            item["answer"] = "no"
        item["question"] = item["input"]
        examples.append(item)
    print(f"{len(examples)} examples")
    return examples


def read_file_coinflip(file_path="../dataset/SymbolicReasoning/CoinFlip/coin_flip.json"):
    json_data = {}
    examples = []
    with open(file_path, "r", encoding="utf8") as f:
        json_data = json.load(f)
    for item in json_data["examples"]:
        item["Question"] = item["question"]
        item["correct"] = item["answer"]
        examples.append(item)
    print(f"{len(examples)} examples")
    return examples


def read_file_lastletters(file_path="../dataset/SymbolicReasoning/LastLetterConcatenation/last_letters.json"):
    json_data = {}
    examples = []
    with open(file_path, "r", encoding="utf8") as f:
        json_data = json.load(f)
    for item in json_data["examples"]:
        item["Question"] = item["question"]
        item["correct"] = item["answer"]
        examples.append(item)
    print(f"{len(examples)} examples")
    return examples


def read_file_date(file_path="../dataset/CommonsenseReasoning/Date/task.json"):
    json_data = {}
    examples = []
    with open(file_path, "r", encoding="utf8") as f:
        json_data = [eval(line) for line in f.readlines()]
    for item in json_data:
        item["question"] = item["input"]
        ans = item["target_scores"]
        for k in ans:
            if ans[k] == 1:
                item["correct"] = k.strip()
                item["answer"] = k.strip()
        examples.append(item)
    print(f"{len(examples)} examples")
    return examples


def write_model_out(file_path, json_data):
    with open(file_path, "a", encoding="utf8") as f:
        f.write(str(json_data))
        f.write("\n")


def get_answer_mgsm_cot(answer: str) -> str:
    # answer = answer.strip().split('he answer is')[-1]

    # match = re.search(r'([-+]?[0-9]*\.?[0-9]+)', answer)
    # if match:
    #     answer = match.group(1)
    # else:
    #     answer = 'NONE'
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
        # if "=" in answer:
        #     answer = answer.split("=")[-1].strip()
        # if "dollors" in answer:
        #     answer = answer.replace("dollars","").strip()
        answer = re.findall(r'\d+', answer)
        print(answer)
        if answer:
            return answer[0]
    return answer


def get_answer_mgsm_standard(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
        try:
            answer = re.findall(r'\d+', answer)[-1]
        except:
            answer = "None"
    return answer


def get_answer_mgsm_contrastive(answer: str) -> str:
    pattern = re.compile(r'Answer: (.*?)\n')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_cleft_sentences(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_nested_clauses(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_fronting(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_nominalization(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_passive(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_relevance_coherence(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_relevance_coherence_invalid(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_bridge_objects_no_coherence(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_bridge_objects_no_relevance(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_language_template_no_coherence(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_language_template_no_relevance(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_no_relevance(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_no_coherence(answer: str) -> str:
    pattern = re.compile(r'he answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_mgsm_step_1(answer: str) -> str:
    return None


def get_answer_mgsm_step_2(answer: str) -> str:
    return None


def get_answer_mgsm_step_3(answer: str) -> str:
    return None


def get_answer_aqua(answer: str) -> str:
    answer = answer.strip().split('he answer is ')[-1]
    # remove punctuation
    processed_string = ''.join(char for char in answer if char.isalnum()).upper()

    return processed_string


def get_answer_aqua_standard(answer: str) -> str:
    pattern1 = re.compile(r'answer is \((.*?)\)\.')
    pattern2 = re.compile(r'answer is \((.*?)\)')
    match1 = pattern1.search(answer.strip())
    match2 = pattern2.search(answer.strip())
    pre = "None"
    if match1:
        pre = match1.group(1).strip().upper()
    elif match2:
        pre = match2.group(1).strip().upper()
    return pre


def get_answer_aqua_cot(answer: str) -> str:
    pattern1 = re.compile(r'answer is \((.*?)\)\.')
    pattern2 = re.compile(r'answer is \((.*?)\)')
    match1 = pattern1.search(answer.strip())
    match2 = pattern2.search(answer.strip())
    pre = "None"
    if match1:
        pre = match1.group(1).strip().upper()
    elif match2:
        pre = match2.group(1).strip().upper()
    return pre


def get_answer_bamboogle(answer: str) -> str:
    pattern = re.compile(r'answer is \s*\((.*?)\)\s*is:\s*(.*?)\.')
    match = pattern.search(answer)
    answer = ""
    if match:
        answer = match.group(2).strip()
    return answer


def get_answer_bamboogle_standard(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer.strip())
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_bamboogle_cot(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer.strip())
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_bamboogle_relevance_coherence(answer: str) -> str:
    pattern = re.compile(r'\) is: (.*?)\n')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
        if answer.endswith("."):
            answer = answer[:-1]
    return answer


def get_answer_bamboogle_relevance_coherence_invalid(answer: str) -> str:
    pattern = re.compile(r'\) is: (.*?)\n')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
        if answer.endswith("."):
            answer = answer[:-1]
    return answer


def get_answer_bamboogle_bridge_objects_no_coherence(answer: str) -> str:
    pattern = re.compile(r'\) is: (.*?)\n')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
        if answer.endswith("."):
            answer = answer[:-1]
    return answer


def get_answer_bamboogle_bridge_objects_no_relevance(answer: str) -> str:
    pattern = re.compile(r'\) is: (.*?)\n')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
        if answer.endswith("."):
            answer = answer[:-1]
    return answer


def get_answer_bamboogle_language_template_no_relevance(answer: str) -> str:
    pattern = re.compile(r'\) is: (.*?)\n')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
        if answer.endswith("."):
            answer = answer[:-1]
    return answer


def get_answer_bamboogle_language_template_no_coherence(answer: str) -> str:
    pattern = re.compile(r'\) is: (.*?)\n')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
        if answer.endswith("."):
            answer = answer[:-1]
    return answer


def get_answer_bamboogle_no_coherence(answer: str) -> str:
    pattern = re.compile(r'\) is: (.*?)\n')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
        if answer.endswith("."):
            answer = answer[:-1]
    return answer


def get_answer_bamboogle_no_relevance(answer: str) -> str:
    pattern = re.compile(r'\) is: (.*?)\n')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
        if answer.endswith("."):
            answer = answer[:-1]
    return answer


def get_answer_sports(answer: str) -> str:
    pattern = re.compile(r'So the answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    if "YES" == answer.upper():
        return "plausible"
    if "NO" == answer.upper():
        return "implausible"
    return "None"


def get_answer_sports(answer: str) -> str:
    pattern = re.compile(r'A: (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_sports_standard(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_sports_cot(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_sports_fronting(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_sports_nested_clauses(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_sports_cleft_sentences(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_sports_passive(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_sports_nominalization(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_coinflip(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    if "YES" == answer.upper():
        return "yes"
    if "NO" == answer.upper():
        return "no"
    return "None"


def get_answer_coinflip_cot(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    if "YES" == answer.upper():
        return "yes"
    if "NO" == answer.upper():
        return "no"
    return "None"


def get_answer_coinflip_step(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    if "YES" == answer.upper():
        return "yes"
    if "NO" == answer.upper():
        return "no"
    return "None"


def get_answer_date(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_date_standard(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_date_cot(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_date_nomonalization(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_date_passive(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_date_cleft(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_date_fronting(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_date_nested(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_lastletters(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_lastletters_cot(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_lastletters_step(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer


def get_answer_strategyqa(answer: str) -> str:
    pattern = re.compile(r'So the answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    if "YES" == answer.upper():
        return "yes"
    if "NO" == answer.upper():
        return "no"
    return "None"


def get_answer_strategyqa_standard(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    if "YES" == answer.upper():
        return "yes"
    if "NO" == answer.upper():
        return "no"
    return "None"


def get_answer_strategyqa_cot(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    if "YES" == answer.upper():
        return "yes"
    if "NO" == answer.upper():
        return "no"
    return "None"


def get_answer_strategyqa_step(answer: str) -> str:
    pattern = re.compile(r'answer \(Yes or No\) is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    if "YES" == answer.upper():
        return "yes"
    if "NO" == answer.upper():
        return "no"
    return "None"


def get_answer_svamp(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    pattern2 = re.compile(r'answer is (.*?)')
    match = pattern.search(answer)
    match2 = pattern2.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    elif match2:
        answer = match2.group(1).strip()
    return answer


def get_answer_svamp_standard(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    pattern2 = re.compile(r'answer is (.*?)')
    match = pattern.search(answer)
    match2 = pattern2.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    elif match2:
        answer = match2.group(1).strip()
    return answer


def get_answer_svamp_cot(answer: str) -> str:
    pattern = re.compile(r'answer is (.*?)\.')
    pattern2 = re.compile(r'answer is (.*?)')
    match = pattern.search(answer)
    match2 = pattern2.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    elif match2:
        answer = match2.group(1).strip()
    return answer


def get_answer_svamp_step(answer: str) -> str:
    pattern = re.compile(r'answer \(arabic numerals\) is (.*?)\.')
    match = pattern.search(answer)
    answer = "None"
    if match:
        answer = match.group(1).strip()
    return answer
