import os
import json
import pandas as pd
from datasets import load_dataset

# Create folders if necessary
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def save_json(data, folder_path, file_name):
    complete_path = os.path.join(BASE_DIR, folder_path)
    os.makedirs(complete_path, exist_ok=True)
    with open(os.path.join(complete_path, file_name), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved: {folder_path}/{file_name} ({len(data)} documents)")

def process_squad():
    print("Download SQuAD 2.0 (Validation split)...")
    dataset = load_dataset("squad_v2", split="validation")
    df = dataset.to_pandas()
    
    # Group info by context
    groups = df.groupby('context')
    
    processed_data = []
    
    for context, group in groups:
        questions = ""
        ground_truths = {}
        
        # Questions of each context
        for i, row in enumerate(group.itertuples()):
            key = f"Q{i+1}" # General keys: Q1, Q2, Q3...
            questions += f"{key}: {row.question}\n"
            
            # Extract answer if exists. Some questions may not have answer
            answers = row.answers['text']
            correct_answer = answers[0] if len(answers) > 0 else "No answer found"
            
            ground_truths[key] = correct_answer
            
        processed_data.append({"text": context,"questions": questions.strip(),"ground_truths": ground_truths })
        # ground truths: Dict[str, str] --> key:ground truth

    save_json(processed_data, "squad", "parsed_squad.json")

def process_newsqa():
    print("Download NewsQA (Validation split) via MRQA...")
    dataset_mrqa = load_dataset("mrqa", split="validation")
    df_mrqa = dataset_mrqa.to_pandas()
    df_filter = df_mrqa[df_mrqa['subset'] == 'NewsQA']
    
    # Group info by context
    groups = df_filter.groupby('context')

    processed_data = []

    for context, group in groups:
        questions = ""
        ground_truths = {}

        # Questions of each context
        for i, row in enumerate(group.itertuples()):
            key = f"Q{i+1}"# General keys: Q1, Q2, Q3...
            questions += f"{key}: {row.question}\n"

            # Extract answer if exists. Some questions may not have answer
            answers = row.answers
            correct_answer = answers[0] if len(answers) > 0 else "No answer found"
            ground_truths[key] = correct_answer
            
        processed_data.append({"text": context, "questions": questions.strip(), "ground_truths": ground_truths})
        # ground truths: Dict[str, str] --> key:ground truth

    save_json(processed_data, "newsqa", "parsed_newsqa.json")

def process_triviaqa():
    print("Download TriviaQA (Validation split)")
    dataset = load_dataset("trivia_qa", "rc", split="validation")
    df = dataset.to_pandas()
    
    data_dict = {}
    
    # TriviaQA groups as (1 question -> several texts). Change it to (1 text -> several questions)
    for row in df.itertuples():
        question = row.question
        # TriviaQA has several possible answers. We take the main (normalized)
        correct_answer = row.answer['normalized_value']
        
        # Extract the context
        wikipedia = row.entity_pages['wiki_context']
        if len(wikipedia) > 0:
            context = wikipedia[0]
            
            if context not in data_dict:
                data_dict[context] = {"q": [], "a": []}
                
            data_dict[context]["q"].append(question)
            data_dict[context]["a"].append(correct_answer)

    processed_data = []

    for context, data in data_dict.items():
        questions = ""
        ground_truths = {}

        # Questions of each context
        for i, (q, a) in enumerate(zip(data["q"], data["a"])):
            key = f"Q{i+1}" # General keys: Q1, Q2, Q3...
            questions += f"{key}: {q}\n"

            ground_truths[key] = a
            
        processed_data.append({"text": context, "questions": questions.strip(), "ground_truths": ground_truths})
        # ground truths: Dict[str, str] --> key:ground truth

    save_json(processed_data, "triviaqa", "parsed_triviaqa.json")

def process_natural_questions_mrqa():
    print("Download Natural Questions (Validation split) vía MRQA...")
    dataset_mrqa = load_dataset("mrqa", split="validation")
    df_mrqa = dataset_mrqa.to_pandas()
    df_filter = df_mrqa[df_mrqa['subset'] == 'NaturalQuestionsShort']
    
    # Group info by context
    groups = df_filter.groupby('context')

    processed_data = []

    for context, group in groups:
        questions = ""
        ground_truths = {}

        # Questions of each context
        for i, row in enumerate(group.itertuples()):
            key = f"Q{i+1}" # General keys: Q1, Q2, Q3...
            questions += f"{key}: {row.question}\n"

            # Extract answer if exists. Some questions may not have answer
            answers = row.answers
            correct_answer = answers[0] if len(answers) > 0 else "No answer found"
            
            ground_truths[key] = correct_answer
            
        processed_data.append({"text": context, "questions": questions.strip(), "ground_truths": ground_truths})
        # ground truths: Dict[str, str] --> key:ground truth
    
    save_json(processed_data, "natural_questions", "parsed_naturalquestions.json")

if __name__ == "__main__":

    process_squad()
    process_newsqa()
    process_triviaqa()
    process_natural_questions_mrqa()
    
    print("Datasets downloaded")