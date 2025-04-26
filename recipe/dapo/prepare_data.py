from datasets import load_dataset
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
from verl.utils.reward_score.math import *
from verl.utils.reward_score.math_dapo import normalize_final_answer
import json
def load_json(path):
    with open (path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def extract_answer(solution,extract_type="TEST"):
    # verl/verl/utils/reward_score/__init__.py
    answer=""
    if extract_type == "think":
        return {
            "ground_truth": solution# use raw solution and extract answer later in reward model
        }
   
    string_in_last_boxed = last_boxed_only_string(solution)
    if string_in_last_boxed is not None:
        if extract_type == "TEST":  # extract ground truth for math compute score
            answer = remove_boxed(string_in_last_boxed)
        elif extract_type == "dapo":  # extract ground truth for math_dapo compute score
            answer = normalize_final_answer(string_in_last_boxed)
    return {
        "ground_truth": answer
        }
def prepare_question(question,prompt_type="TEST"):
    prompt_list = []
    # Escape curly braces to avoid conflicts with format method
    if prompt_type == "dapo":
        user_prompt = f"Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{question}\n\nRemember to put your answer on its own line after \"Answer:\""
        prompt_list.append(
            {"content": user_prompt, "role": "user"}
        )
    elif prompt_type == "think":
        system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>." 
        prompt_list.append(
            {"content":system_prompt, "role": "system"}
        )
        prompt_list.append(
            {"content": question, "role": "user"}
        )
    elif prompt_type == "TEST":
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."
        prompt_list.append(
            {"content":system_prompt, "role": "system"}
        )
        prompt_list.append(
            {"content": question, "role": "user"}
        )

    return prompt_list

if __name__ == "__main__":
    data_source = "think"
    # 加载数据集的 train 部分
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    save_path = f"/data2/xucaijun/verl/data/{data_source}_MATH-lighteval_train-processed.parquet"
    # dataset = load_dataset("HuggingFaceH4/MATH-500",split="test")
    # save_path = f"/data2/xucaijun/verl/data/{data_source}_MATH-500-processed.parquet"
    # dataset = load_json("/data2/xucaijun/My-Math-Generator/outputs/7b_rejected_first_filter_1-4.json")
    # save_path = f"/data2/xucaijun/verl/data/{data_source}_7b_rejected_first_filter_1-4.parquet"
    processed_data = []

    for i,item in enumerate(dataset):
        question = item['problem']  # Extract the problem
        solution = item['solution']  # Extract the solution
        
        # Generate the prompt and answer
        prompt = prepare_question(question,prompt_type=data_source)
        answer = extract_answer(solution,extract_type=data_source)
        if i ==0 :
            print(f"Prompt: {prompt}")
            print(f"Answer: {answer}")
        if len(answer['ground_truth'])==0:
            print(f"Error: No ground truth found for solution: {solution}")
            continue
        # Append to the processed data
        processed_data.append({
            'prompt': prompt,
            'reward_model': answer,
            "data_source": data_source,
            "extra_info":{"index": i}
        })

    processed_df = pd.DataFrame(processed_data)
    processed_df.to_parquet(save_path)

    print(f"Processed {len(processed_data)} dataset saved as Parquet.")
