from core.preprocessing import preprocess, preprocess_questions, parse_questions_embeddings, split_text_into_sentences

def warmup_preprocessing_cache(text: str, questions: dict):
    """Executes preprocessing to store data in cache and avoid the first models having unjust computation times"""
    preprocess(text)
    preprocess_questions(questions)
    parse_questions_embeddings(questions)
    split_text_into_sentences(text)