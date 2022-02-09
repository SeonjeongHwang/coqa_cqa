import json
import pandas as pd
import tqdm
import copy

from tool.process import *


class Example(object):    
    def __init__(self,
                 qas_id,
                 doc_tokens,
                 history,
                 question_text,
                 answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.history = history
        self.question_text = question_text
        self.answer_text = answer_text
        self.start_position = start_position
        self.end_position = end_position
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += ", doc_tokens: [%s]\n" % (" ".join(self.doc_tokens))
        s += f"history: {self.history}\n"
        s += ", question: %s\n" % (self.question_text)
        if self.answer_text:
            s += ", answer: %s\n" % (self.answer_text)
        if self.start_position:
            s += ", start_position: %d\n" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d\n" % (self.end_position)
        return s

def read_example(input_file, is_training, num_turn):
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for idx, data in tqdm.tqdm(enumerate(input_data), total=len(input_data)):
        data_id = data["id"]
        paragraph_text = data["story"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)    
    
        questions = sorted(data["questions"], key=lambda x: x["turn_id"])
        answers = sorted(data["answers"], key=lambda x: x["turn_id"])
        
        history = []
        qas = list(zip(questions, answers))
        for i, (question, answer) in enumerate(qas):
            qas_id = "{0}_{1}".format(data_id, i+1)
        
            question_text = question["input_text"]
            answer_type, answer_subtype = get_answer_type(question, answer)
            answer_text, span_start, span_end, is_skipped, _, _, _ = get_answer_span(answer, answer_type, paragraph_text)
        
            if answer_type in ["yes", "no", "unknown"]:
                if answer_type != "unknown":
                    history.append((question_text, answer["input_text"]))
                continue
            
            start_position = char_to_word_offset[span_start]
            end_position = char_to_word_offset[span_end]

            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            
            def whitespace_tokenize(text):
                """Runs basic whitespace cleaning and splitting on a piece of text."""
                text = text.strip()
                if not text:
                    return []
                tokens = text.split()
                return tokens
            
            actual_text = " ".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                whitespace_tokenize(answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
                print("Could not find answer: "+actual_text+" vs. "+cleaned_answer_text)
                continue
            
            example = Example(qas_id=qas_id,
                              doc_tokens=doc_tokens,
                              history=history[-num_turn:].copy(),
                              question_text=question_text,
                              answer_text=answer_text,
                              start_position=start_position,
                              end_position=end_position)
            
            history.append((question_text, answer["input_text"]))
            examples.append(example)
           
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_history_length, is_training):

    unique_id = 1000000000
    features = {"unique_id":[],
                "example_index":[],
                "doc_span_index":[],
                "tokens":[],
                "token_to_orig_map":[],
                "token_is_max_context":[],
                "input_ids":[],
                "attention_mask":[],
                "segment_ids":[],
                "start_position":[],
                "end_position":[],
               }

    ## [CLS] <Q> Q <A> A <Q> Q <A> A <Q> Q [SEP] document [SEP]
    for (example_index, example) in tqdm.tqdm(enumerate(examples), total=len(examples)):
        history_input_ids = []
        history_tokens = []
        for q, a in example.history:
            history_input_ids.append(tokenizer.encode("<Q>")[1])
            history_input_ids.extend(tokenizer.encode(q)[1:-1])
            history_input_ids.append(tokenizer.encode("<A>")[1])
            history_input_ids.extend(tokenizer.encode(a)[1:-1])
            
            history_tokens.append("<Q>")
            history_tokens.extend(tokenizer._tokenize(q))
            history_tokens.append("<A>")
            history_tokens.extend(tokenizer._tokenize(a))
            
        if len(history_input_ids) > max_history_length:
            history_input_ids = history_input_ids[-max_history_length:]
            history_tokens = history_tokens[-max_history_length:]
            
        assert len(history_input_ids) == len(history_tokens)
        
        question_input_ids = tokenizer.encode("<Q>")[1:-1] + tokenizer.encode(example.question_text)[1:-1] 
        question_tokens = ["<Q>"] + tokenizer._tokenize(example.question_text)

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for i, token  in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer._tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.answer_text)    

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(history_tokens) - len(question_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            input_ids = []
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            
            input_ids.append(tokenizer.cls_token_id)
            tokens.append(tokenizer.cls_token)
            segment_ids.append(0)
            
            input_ids.extend(history_input_ids)
            tokens.extend(history_tokens)
            segment_ids.extend([0]*len(history_tokens))
            
            input_ids.extend(question_input_ids)
            tokens.extend(question_tokens)
            segment_ids.extend([0]*len(question_tokens))
            
            input_ids.append(tokenizer.sep_token_id)
            tokens.append(tokenizer.sep_token)
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
  
                is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                                 split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                input_ids.append(tokenizer._convert_token_to_id(all_doc_tokens[split_token_index]))
                segment_ids.append(1)
                
            input_ids.append(tokenizer.sep_token_id)
            tokens.append(tokenizer.sep_token)
            segment_ids.append(1)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(tokenizer.pad_token_id)
                attention_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(history_tokens) + len(question_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
 
            if example_index < 5:
                print("*** Example ***")
                print("unique_id: %s" % (unique_id))
                print("example_index: %s" % (example_index))
                print("tokens: %s" % " ".join(tokens))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print(
                    "attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                print(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                print("answer_text: %s" % (example.answer_text))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    print("span answer: %s" % (answer_text))
                    print("start_position: %d" % (start_position))
                    print("end_position: %d" % (end_position))
            
            features["unique_id"].append(unique_id)
            features["example_index"].append(example_index)
            features["doc_span_index"].append(doc_span_index)
            features["tokens"].append(tokens)
            features["token_to_orig_map"].append(token_to_orig_map)
            features["token_is_max_context"].append(token_is_max_context)
            features["input_ids"].append(input_ids)
            features["attention_mask"].append(attention_mask)
            features["segment_ids"].append(segment_ids)
            features["start_position"].append(start_position)
            features["end_position"].append(end_position)

            unique_id += 1
               
    return features

def save_features_as_pkl(features, file_name):
    temp = pd.DataFrame(features)
    temp.to_pickle(file_name)
    
def load_features_from_pkl(file_name):
    data = pd.read_pickle(file_name).to_dict()
    features = dict()
    for key, values in data.items():
        features[key] = list(values.values())
    return features