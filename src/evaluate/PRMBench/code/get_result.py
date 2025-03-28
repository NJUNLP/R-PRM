import json
import os
import argparse
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# from mr_eval.utils.utils import *
prm_model_name_dict = dict(
    skyworkprm_1_5B="\\href{https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B}{Skywork-PRM-1.5B}",
    skyworkprm_7B="\\href{https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B}{Skywork-PRM-7B}",
    llemma7b_prm_prm800k="\\href{https://huggingface.co/ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf}{Llemma-PRM800k-7B}",
    llemma7b_prm_metamath="\\href{https://huggingface.co/ScalableMath/llemma-7b-prm-metamath-level-1to3-hf}{Llemma-MetaMath-7B}",
    llemma7b_oprm_prm800k="\\href{https://huggingface.co/ScalableMath/llemma-7b-oprm-prm800k-level-1to3-hf}{Llemma-oprm-7B}",
    mathminos_mistral="\\href{https://github.com/KbsdJames/MATH-Minos}{MATHMinos-Mistral-7B}",
    mathshepherd="\\href{https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm}{MathShepherd-Mistral-7B}",
    reasoneval7b="\\href{https://huggingface.co/GAIR/ReasonEval-7B}{ReasonEval-7B}",
    llama3_1_8b_prm_mistral="\\href{https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Mistral-Data}{RLHFlow-PRM-Mistral-8B}",
    llama3_1_8b_prm_deepseek="\\href{https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data}{RLHFlow-PRM-Deepseek-8B}",
    reasoneval34b="\\href{https://huggingface.co/GAIR/ReasonEval-34B}{ReasonEval-34B}",
)
classification_name_dict = dict(
    domain_inconsistency="DC.",
    redundency="NR.",
    multi_solutions="MS.",
    deception="DR.",
    confidence="CI.",
    step_contradiction="SC.",
    circular="NCL.",
    missing_condition="PS.",
    counterfactual="ES."
)
classification_parallel_dict = dict(
    simplicity=dict(
        redundency="NR.",
        circular="NCL.",
    ),
    soundness=dict(
        counterfactual="ES.",
        step_contradiction="SC.",
        domain_inconsistency="DC.",
        confidence="CI.",
    ),
    sensitivity=dict(
        missing_condition="PS.",
        deception="DR.",
        multi_solutions="MS.",
    )
)

def list_jsonl_files(folder_path):
    """
    列举文件夹中的所有 .jsonl 文件
    Args:
        folder_path (str): 文件夹路径
    Returns:
        List[str]: 所有 .jsonl 文件的路径
    """
    return [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]

def process_jsonl(file_path):
    '''
        将jsonl文件转换为装有dict的列表
    '''
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

def get_res_dict(file_dict, model_name, model_lists=None):
    # import code; code.interact(local=locals())
    res_dict = {}
    file_path = file_dict[model_name]
    res_dict[model_name] = process_jsonl(file_path)[-1]
    return res_dict

def get_prmscore_from_current_res_dict(res_dict,classification=None):
    '''
    Get PRM score from model level dict
    '''
    if not classification:
        prm_score = res_dict["total_hallucination_results"]['f1'] * 0.5 + res_dict["total_hallucination_results"]['negative_f1'] * 0.5
    else:
        if classification in ["multi_solutions"]:
            prm_score = res_dict["hallucination_type_results"]['f1'][classification]
        else:
            prm_score = res_dict["hallucination_type_results"]['f1'][classification] * 0.5 + res_dict["hallucination_type_results"]['negative_f1'][classification] * 0.5
    return prm_score


def get_avg_prmscore_from_current_res_dict(res_dict,classifications):
    '''
    Get AVG PRM score from model level dict
    '''
    assert classifications
    res = [get_prmscore_from_current_res_dict(res_dict,classification) for classification in classifications]
    return sum(res) / len(res)
    

def get_res_str(model_name, classification_dict,res_dict):
    res_str = ""
    # current_classification_dict = classification_dict[classification_name]
    avg_res_list = []
    # for idx,(model_name, model_display_name) in enumerate(model_dict.items()):
    temp_str = "| Model | Overall| NR. | NCL. | Avg (simplicity) | ES. | SC. | DC. | CI. | Avg (soundness) | PS. | DR. | MS. | Avg (sensitivity)  |\n"
    temp_str += f"{model_name}"
    current_res_dict = res_dict[model_name]
    prm_score = get_prmscore_from_current_res_dict(current_res_dict)
    all_model_scores = sorted([get_prmscore_from_current_res_dict(res) for res in res_dict.values()],reverse=True)
    if prm_score == max(all_model_scores):
        temp_str += f" |{prm_score * 100:.1f}|"
    elif prm_score == all_model_scores[1]:
        temp_str += f" & \\underline{{{prm_score * 100:.1f}}}"
    else:
        temp_str += f" |{prm_score * 100:.1f}|"
    
    for big_classification, current_classification_dict in classification_dict.items():
        all_avt = sorted([get_avg_prmscore_from_current_res_dict(res,list(current_classification_dict.keys())) for res in res_dict.values()], reverse=True)
        avg = []
        for classification, display_classification_name in current_classification_dict.items():
            prm_score = get_prmscore_from_current_res_dict(current_res_dict,classification)
            all_prm_scores = sorted([get_prmscore_from_current_res_dict(res,classification) for res in res_dict.values()], reverse=True)
            avg.append(prm_score)
            if prm_score == max(all_prm_scores):
                temp_str += f" \t|{prm_score * 100:.1f}"
            elif prm_score == all_prm_scores[1]:
                temp_str += f" & \\underline{{{prm_score * 100:.1f}}}"
            else:
                temp_str += f" \t| {prm_score * 100:.1f}"
        avg_score = sum(avg) / len(avg)
        if avg_score == max(all_avt):
            temp_str += f" \t|{avg_score * 100:.1f}"
        elif avg_score == all_avt[1]:
            temp_str += f" & \\underline{{{avg_score * 100:.1f}}}"
        else:
            temp_str += f" \t|{avg_score * 100:.1f}"
    temp_str += "\\\\\n"
    res_str += temp_str
    
    return res_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='.', help='The directory of the result files')
    args = parser.parse_args()
    
    classifications = ["redundency", "circular", "counterfactual", "step_contradiction", "domain_inconsistency",  "confidence", "missing_condition", "deception", "multi_solutions", ]
    metrics = ["f1", "negative_f1", "total_step_acc", "correct_step_acc", "wrong_step_acc", "first_error_acc", "similarity",]

    ## File paths
    res_dir = "../data"
    res_files = list_jsonl_files(res_dir)
    res_names = [f.split(".")[0] for f in res_files]
    res_paths = [os.path.join(res_dir, f) for f in res_files]
    file_dict = dict(zip(res_names, res_paths))

    res_dict = get_res_dict(file_dict, f'{args.model_name}', model_lists=list(prm_model_name_dict.keys()))
    prm_str = get_res_str(f'{args.model_name}', classification_parallel_dict, res_dict)
    res_str = ""
    res_str += prm_str
    print(res_str)
# import code; code.interact(local=locals())