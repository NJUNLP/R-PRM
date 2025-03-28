import json
import random
import argparse
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def eval_on_hallucination_step(hallucination_steps, labels, redundency_label=False):
    ## Important: hallucination_steps are 0-indexed
    hallucination_steps = [i-1 for i in hallucination_steps]
    ## Important: hallucination_steps are 0-indexed
    if redundency_label:
        POSITIVE_LABEL = 0
        NEGATIVE_LABEL = 1
    else:
        POSITIVE_LABEL = 1
        NEGATIVE_LABEL = 0

    correct_step_acc = []
    wrong_step_acc = []
    total_step_acc = []
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    first_error_location = min(hallucination_steps) if len(hallucination_steps)>0 else -1
    first_error_acc = None
    for idx in range(len(labels)):
        
        if idx == first_error_location:
            if labels[idx] == NEGATIVE_LABEL:
                first_error_acc = 1
            else:
                first_error_acc = 0
        
        if idx in hallucination_steps:
            if labels[idx] == POSITIVE_LABEL:
                wrong_step_acc.append(0)
                total_step_acc.append(0)
                FP += 1
            else:
                wrong_step_acc.append(1)
                total_step_acc.append(1)
                TN += 1
        else:
            if labels[idx] == POSITIVE_LABEL:
                correct_step_acc.append(1)
                total_step_acc.append(1)
                TP += 1
            else:
                correct_step_acc.append(0)
                total_step_acc.append(0)
                FN += 1
                
    
    correct_step_acc_value = sum(correct_step_acc)/len(correct_step_acc) if len(correct_step_acc)>0 else -1
    wrong_step_acc_value = sum(wrong_step_acc)/len(wrong_step_acc) if len(wrong_step_acc)>0 else -1
    total_step_acc_value = sum(total_step_acc)/len(total_step_acc) if len(total_step_acc)>0 else -1
    model_response_acc = sum(labels)/len(labels) if len(labels)>0 else -1
    
    return dict(
        correct_step_acc=correct_step_acc_value,
        wrong_step_acc=wrong_step_acc_value,
        total_step_acc=total_step_acc_value,
        first_error_acc=first_error_acc,
        model_response_acc=model_response_acc,
        f1_matrix = dict(TP=TP,FP=FP,TN=TN,FN=FN),
        
        
        correct_step_acc_list=correct_step_acc,
        wrong_step_acc_list=wrong_step_acc,
        total_step_acc_list=total_step_acc,
        first_error_acc_list=[first_error_acc] if first_error_acc is not None else [],
        model_response_acc_list=[model_response_acc] if model_response_acc != -1 else [],
    )

def evaluate_function(results,meta_data):
    meta_data_dict = {meta["idx"]: meta for meta in meta_data}
    classification_types = set([meta["classification"] for meta in meta_data])
    metric_types = ["correct_step_acc","wrong_step_acc","total_step_acc","first_error_acc"]
    halucination_specified_dict = {}
    total_metric_lists = {}
    for metric in metric_types+["similarity"]:
        halucination_specified_dict[metric] = {i:[] for i in classification_types}
        total_metric_lists[metric] = []
    halucination_specified_dict["f1_matrix"] = {i:dict(TP=0,FP=0,TN=0,FN=0) for i in classification_types}
    total_metric_lists["f1_matrix"] = dict(TP=0,FP=0,TN=0,FN=0)
    

    detailed_logs = []
    valid_num = 0
    total_num = len(meta_data)
    
    ## Filter out repeated items
    filtered_dict = {}
    filtered_results = []
    for result in results:
        idx = result["idx"]
        if filtered_dict.get(idx) is None and meta_data_dict.get(idx) is not None:
            filtered_dict[idx] = 1
            filtered_results.append(result)
    assert abs(len(filtered_results) - len(meta_data)) < 5, f"filtered_results number: {len(filtered_results)}, meta_data number: {len(meta_data)}"

    correct_ids_dict = {meta["idx"]:1 for meta in meta_data if meta["classification"] == "correct"} 
    correct_predictions  = [prediction for prediction in filtered_results if prediction["idx"] in correct_ids_dict]
    other_predictions = [prediction for prediction in filtered_results if prediction["idx"] not in correct_ids_dict]
    correct_model_response_acc_dict = {}
    
    ## First evaluate the correct samples
    for prediction in correct_predictions:
        idx = prediction["idx"]
        reference_item = meta_data_dict[idx]
        error_steps = reference_item["error_steps"]    
        classifcation = reference_item["classification"]
        assert classifcation == "correct"
        
        if "validity" in prediction and not prediction["validity"]:
            log = dict(
                idx=idx,
                error_steps=error_steps,
                classifcation=classifcation,
                prediction=None,
                results=None,
                )
        else:
            labels = prediction["scores"]["step_level_validity_labels"] if "step_level_validity_labels" in prediction["scores"] else None
            res_dict = eval_on_hallucination_step(error_steps,labels)
                
            for metric in metric_types:
                # total_metric_lists[metric].extend(res_dict[f'{metric}_list'])
                halucination_specified_dict[metric][classifcation].extend(res_dict[f'{metric}_list'])
            halucination_specified_dict["f1_matrix"][classifcation]["TP"] += res_dict["f1_matrix"]["TP"]
            halucination_specified_dict["f1_matrix"][classifcation]["FP"] += res_dict["f1_matrix"]["FP"]
            halucination_specified_dict["f1_matrix"][classifcation]["TN"] += res_dict["f1_matrix"]["TN"]
            halucination_specified_dict["f1_matrix"][classifcation]["FN"] += res_dict["f1_matrix"]["FN"]
            
            correct_model_response_acc_dict[idx] = res_dict["model_response_acc"]   
            log = dict(
                idx=idx,
                error_steps=error_steps,
                classifcation=classifcation,
                prediction=prediction,
                results=res_dict,
            )
            detailed_logs.append(log)
    
    ## Then evaluate the other sample types
    for prediction in other_predictions:
        idx = prediction["idx"]
        
        if "validity" in prediction and not prediction["validity"]:
            log = dict(
                idx=idx,
                hallucination_steps=None,
                hallucination_types=None,
                prediction=None,
                results=None,
                validitiy=False,
                )
        else:
            valid_num += 1
            try:
                reference_item = meta_data_dict[idx]
            except:
                logger.info(f"idx {idx} not found in meta_data_dict")
                continue
            error_steps = reference_item["error_steps"]
            classifcation = reference_item["classification"]
            
            if (classifcation == "redundency" or classifcation == "circular") and "step_level_redundancy_labels" in prediction["scores"]:
                labels = prediction["scores"]["step_level_redundancy_labels"]
                labels = [ not i for i in labels]
                res_dict = eval_on_hallucination_step(error_steps,labels,redundency_label=False)
            else:
                labels = prediction["scores"]["step_level_validity_labels"] if "step_level_validity_labels" in prediction["scores"] else None
                res_dict = eval_on_hallucination_step(error_steps,labels,redundency_label=False)
                
            for metric in metric_types:
                total_metric_lists[metric].extend(res_dict[f'{metric}_list'])
                halucination_specified_dict[metric][classifcation].extend(res_dict[f'{metric}_list'])
            halucination_specified_dict["f1_matrix"][classifcation]["TP"] += res_dict["f1_matrix"]["TP"]
            halucination_specified_dict["f1_matrix"][classifcation]["FP"] += res_dict["f1_matrix"]["FP"]
            halucination_specified_dict["f1_matrix"][classifcation]["TN"] += res_dict["f1_matrix"]["TN"]
            halucination_specified_dict["f1_matrix"][classifcation]["FN"] += res_dict["f1_matrix"]["FN"]
            
            total_metric_lists["f1_matrix"]["TP"] += res_dict["f1_matrix"]["TP"]
            total_metric_lists["f1_matrix"]["FP"] += res_dict["f1_matrix"]["FP"]
            total_metric_lists["f1_matrix"]["TN"] += res_dict["f1_matrix"]["TN"]
            total_metric_lists["f1_matrix"]["FN"] += res_dict["f1_matrix"]["FN"]
            
            correct_idx = "correct_"+idx[len(f"{classifcation}_"):]
            correct_item_acc = correct_model_response_acc_dict.get(correct_idx)
            item_acc = res_dict["model_response_acc"]
            if correct_item_acc and item_acc != -1:
                abs_similarity = abs(item_acc - correct_item_acc)  
                total_metric_lists["similarity"].append(abs_similarity)
                halucination_specified_dict["similarity"][classifcation].append(abs_similarity)
            log = dict(
                        idx=idx,
                        error_steps=error_steps,
                        classifcation=classifcation,
                        prediction=prediction,
                        results=res_dict,
                    )
        detailed_logs.append(log)
    
    
    ## Calculate final results
    total_final_results = {metric:sum(total_metric_lists[metric])/len(total_metric_lists[metric]) if len(total_metric_lists[metric])>0 else -1 for metric in metric_types+["similarity"]}
    hallucination_type_final_results = {metric:{k:sum(v)/len(v) if len(v)>0 else -1 for k,v in halucination_specified_dict[metric].items()} for metric in metric_types+["similarity"]}
    validitiy_rate = valid_num / total_num
    
    
    ## Calculate F1 score
    TP = total_metric_lists["f1_matrix"]["TP"]
    FP = total_metric_lists["f1_matrix"]["FP"]
    FN = total_metric_lists["f1_matrix"]["FN"]
    TN = total_metric_lists["f1_matrix"]["TN"]
    total_precision = TP / (TP + FP) if (TP + FP) != 0 else -1
    total_recall = TP / (TP + FN) if (TP + FN) != 0 else -1
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) != 0 else -1
    negative_precision = TN / (TN + FN) if (TN + FN) != 0 else -1
    negative_recall = TN / (TN + FP) if (TN + FP) != 0 else -1
    negative_f1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall) if (negative_precision + negative_recall) != 0 else -1
    total_final_results["precision"] = total_precision
    total_final_results["recall"] = total_recall
    total_final_results["f1"] = total_f1
    total_final_results["negative_precision"] = negative_precision
    total_final_results["negative_recall"] = negative_recall
    total_final_results["negative_f1"] = negative_f1
    
    for metric in ["precision","recall","f1","negative_precision","negative_recall","negative_f1"]:
        hallucination_type_final_results[metric] = {}
    for classification in classification_types:
        TP = halucination_specified_dict["f1_matrix"][classification]["TP"]
        FP = halucination_specified_dict["f1_matrix"][classification]["FP"]
        FN = halucination_specified_dict["f1_matrix"][classification]["FN"]
        TN = halucination_specified_dict["f1_matrix"][classification]["TN"]
        precision = TP / (TP + FP) if (TP + FP) != 0 else -1
        recall = TP / (TP + FN) if (TP + FN) != 0 else -1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else -1
        negative_precision = TN / (TN + FN) if (TN + FN) != 0 else -1
        negative_recall = TN / (TN + FP) if (TN + FP) != 0 else -1
        negative_f1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall) if (negative_precision + negative_recall) != 0 else -1
        hallucination_type_final_results["precision"][classification] = precision
        hallucination_type_final_results["recall"][classification] = recall
        hallucination_type_final_results["f1"][classification] = f1
        hallucination_type_final_results["negative_precision"][classification] = negative_precision
        hallucination_type_final_results["negative_recall"][classification] = negative_recall
        hallucination_type_final_results["negative_f1"][classification] = negative_f1
    
    res = dict(
        total_hallucination_results=total_final_results,
        hallucination_type_results=hallucination_type_final_results,
        validitiy_rate=validitiy_rate,
        detailed_logs=detailed_logs,
    )
    return res

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

def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str, default="score.jsonl")
    parser.add_argument("--prm_dataset_path", type=str, default="prmbench_preview.jsonl")
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    path1 = args.score_path
    path2 = args.prm_dataset_path
    model_name = args.model_name
    meta_data_path = "../data/meta_data.jsonl"
    dataset1 = load_dataset(path1)
    dataset2 = load_dataset(path2)
    meta_data = load_dataset(meta_data_path)

    cnt = 0
    output_data = []
    for data1, data2 in zip(dataset1, dataset2):
        idx = data2['idx']
        scores = data1['yes_probability_average']
        labels = [score >= 0.61 for score in scores]
        classification = data2['classification']
        dic = {'scores': {'step_level_validity_scores': data1['yes_probability_average'], 'step_level_validity_labels': labels}, 'idx': f'{classification}_{idx}'}
        output_data.append(dic)
    res_log = evaluate_function(output_data, meta_data)

    classifications = ["redundency", "circular", "counterfactual", "step_contradiction", "domain_inconsistency",  "confidence", "missing_condition", "deception", "multi_solutions", ]
    metrics = ["f1", "negative_f1", "total_step_acc", "correct_step_acc", "wrong_step_acc", "first_error_acc", "similarity",]

    res_dict = {}
    res_dict[model_name] = res_log
    prm_str = get_res_str(f'{args.model_name}', classification_parallel_dict, res_dict)
    res_str = ""
    res_str += prm_str
    print(res_str)
