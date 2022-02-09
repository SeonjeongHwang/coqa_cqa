import collections
import re
import six
import math

from tool.eval_coqa import CoQAEvaluator

def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
  
    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
  
    return cur_span_index == best_span_index

def improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer._tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def whitespace_tokenize(text):
    word_spans = []
    char_list = []
    for idx, char in enumerate(text):
        if char != ' ':
            char_list.append(idx)
            continue
            
        if char_list:
            word_start = char_list[0]
            word_end = char_list[-1]
            word_text = text[word_start:word_end+1]
            word_spans.append((word_text, word_start, word_end))
            char_list.clear()
        
    if char_list:
        word_start = char_list[0]
        word_end = char_list[-1]
        word_text = text[word_start:word_end+1]
        word_spans.append((word_text, word_start, word_end))
        
    return word_spans

def find_answer_span(answer_text,
                     rationale_text,
                     rationale_start,
                     rationale_end):
    idx = rationale_text.find(answer_text)
    answer_start = rationale_start + idx
    answer_end = answer_start + len(answer_text) - 1
        
    return answer_start, answer_end

def char_span_to_word_span(char_start,
                            char_end,
                            word_spans):
    word_idx_list = []
    for word_idx, (_, start, end) in enumerate(word_spans):
        if end >= char_start:
            if start <= char_end:
                word_idx_list.append(word_idx)
            else:
                break

    if word_idx_list:
        word_start = word_idx_list[0]
        word_end = word_idx_list[-1]
    else:
        word_start = -1
        word_end = -1

    return word_start, word_end

def search_best_span(context_tokens,
                      answer_tokens):
    best_f1 = 0.0
    best_start, best_end = -1, -1
    search_index = [idx for idx in range(len(context_tokens)) if context_tokens[idx][0] in answer_tokens]
    for i in range(len(search_index)):
        for j in range(i, len(search_index)):
            candidate_tokens = [context_tokens[k][0] for k in range(search_index[i], search_index[j]+1) if context_tokens[k][0]]
            common = collections.Counter(candidate_tokens) & collections.Counter(answer_tokens)
            num_common = sum(common.values())
            if num_common > 0:
                precision = 1.0 * num_common / len(candidate_tokens)
                recall = 1.0 * num_common / len(answer_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_start = context_tokens[search_index[i]][1]
                    best_end = context_tokens[search_index[j]][2]
        
    return best_f1, best_start, best_end

def match_answer_span(answer_text,
                      rationale_start,
                      rationale_end,
                      paragraph_text):
    answer_tokens = whitespace_tokenize(answer_text)
    answer_norm_tokens = [CoQAEvaluator.normalize_answer(token) for token, _, _ in answer_tokens]
    answer_norm_tokens = [norm_token for norm_token in answer_norm_tokens if norm_token]
        
    if not answer_norm_tokens:
        return -1, -1
        
    paragraph_tokens = whitespace_tokenize(paragraph_text)
        
    if not (rationale_start == -1 or rationale_end == -1):
        rationale_word_start, rationale_word_end = char_span_to_word_span(rationale_start, rationale_end, paragraph_tokens)
        rationale_tokens = paragraph_tokens[rationale_word_start:rationale_word_end+1]
        rationale_norm_tokens = [(CoQAEvaluator.normalize_answer(token), start, end) for token, start, end in rationale_tokens]
        match_score, answer_start, answer_end = search_best_span(rationale_norm_tokens, answer_norm_tokens)
            
        if match_score > 0.0:
            return answer_start, answer_end
        
    paragraph_norm_tokens = [(CoQAEvaluator.normalize_answer(token), start, end) for token, start, end in paragraph_tokens]
    match_score, answer_start, answer_end = search_best_span(paragraph_norm_tokens, answer_norm_tokens)
        
    if match_score > 0.0:
        return answer_start, answer_end
        
    return -1, -1

def get_answer_span(answer,
                    answer_type,
                    paragraph_text):
    input_text = answer["input_text"].strip().lower()
    span_start, span_end = answer["span_start"], answer["span_end"]
    if span_start == -1 or span_end == -1:
        span_text = ""
    else:
        span_text = paragraph_text[span_start:span_end].lower()
        
    if input_text in span_text:
        span_start, span_end = find_answer_span(input_text, span_text, span_start, span_end)
    else:
        span_start, span_end = match_answer_span(input_text, span_start, span_end, paragraph_text.lower())
        
    if span_start == -1 or span_end == -1:
        answer_text = ""
        is_skipped = (answer_type == "span")
    else:
        answer_text = paragraph_text[span_start:span_end+1]
        is_skipped = False
        
    span_text = answer["span_text"]
    rat_start = answer["span_start"]
    rat_end = answer["span_end"]
    if span_text == "unknown":
        rat_text = ""
    else:
        rat_text = span_text.lstrip()
        l_white_len = len(span_text)-len(rat_text)
        if l_white_len > 0:
            rat_start += l_white_len
        rat_text = rat_text.rstrip()
        rat_end = rat_start + len(rat_text) - 1

    return answer_text, span_start, span_end, is_skipped, rat_text, rat_start, rat_end

def normalize_answer(answer):
    norm_answer = CoQAEvaluator.normalize_answer(answer)
        
    if norm_answer in ["yes", "yese", "ye", "es"]:
        return "yes"
        
    if norm_answer in ["no", "no not at all", "not", "not at all", "not yet", "not really"]:
        return "no"
        
    return norm_answer

def get_answer_type(question,
                    answer):
    norm_answer = normalize_answer(answer["input_text"])
        
    if norm_answer == "unknown" or "bad_turn" in answer:
        return "unknown", None
        
    if norm_answer == "yes":
        return "yes", None
        
    if norm_answer == "no":
        return "no", None
        
    return "span", None