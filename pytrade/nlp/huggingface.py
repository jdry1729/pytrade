from collections import defaultdict

from nltk.tokenize import sent_tokenize


def get_scores(pipe, text: str, max_token_length: int = 512, tokenizer=None):
    sentences = sent_tokenize(text)

    # tokenizer = AutoTokenizer.from_pretrained(pipe.model.name_or_path)

    scores = []
    current_chunk = ""

    for sentence in sentences:
        if len(tokenizer(current_chunk + " " + sentence)[
                   "input_ids"]) <= max_token_length:
            current_chunk += " " + sentence
        else:
            len_ = len(tokenizer(current_chunk)["input_ids"])
            scores_ = pipe(current_chunk, return_all_scores=True)
            scores.append((scores_[0], len_))
            current_chunk = sentence  # Start a new chunk with the current sentence

    if current_chunk:
        len_ = len(tokenizer(current_chunk)["input_ids"])
        scores_ = pipe(current_chunk, return_all_scores=True)
        scores.append((scores_[0], len_))

    total_len = sum(x for _, x in scores)
    agg_scores = defaultdict(float)
    for scores_, len_ in scores:
        for score in scores_:
            agg_scores[score["label"]] += score["score"] * len_ / total_len

    return agg_scores
