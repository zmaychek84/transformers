import time

import numpy as np
import pandas as pd
import torch

from transformers import BertForQuestionAnswering, BertTokenizer


def data_prep():
    coqa = pd.read_json(
        "http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json"
    )
    coqa.head()

    del coqa["version"]

    # required columns in our dataframe
    cols = ["text", "question", "answer"]
    # list of lists to create our dataframe
    comp_list = []
    for index, row in coqa.iterrows():
        for i in range(len(row["data"]["questions"])):
            temp_list = []
            temp_list.append(row["data"]["story"])
            temp_list.append(row["data"]["questions"][i]["input_text"])
            temp_list.append(row["data"]["answers"][i]["input_text"])
            comp_list.append(temp_list)
    new_df = pd.DataFrame(comp_list, columns=cols)
    # saving the dataframe to csv file for further loading
    new_df.to_csv("CoQA_data.csv", index=False)


def get_question_text(data):
    random_num = np.random.randint(0, len(data))
    question = data["question"][random_num]
    text = data["text"][random_num]
    return question, text


def get_answer(model, tokenizer, question, text, debug=False):
    input_ids = tokenizer.encode(question, text)
    if debug:
        print("The input has a total of {} tokens.".format(len(input_ids)))

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    if debug:
        for token, id in zip(tokens, input_ids):
            print("{:8}{:8,}".format(token, id))

    # first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    if debug:
        print("SEP token index: ", sep_idx)
    # number of tokens in segment A (question) - this will be one more than the sep_idx as the index in Python starts from 0
    num_seg_a = sep_idx + 1
    if debug:
        print("Number of tokens in segment A: ", num_seg_a)
    # number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    if debug:
        print("Number of tokens in segment B: ", num_seg_b)
    # creating the segment ids
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    # making sure that every input token has a segment id
    assert len(segment_ids) == len(input_ids)

    # token input_ids to represent the input and token segment_ids to differentiate our segments - question and text
    start = time.time_ns()
    output = model(
        torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids])
    )
    end = time.time_ns()
    prof = end - start

    # tokens with highest start and end scores
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = " ".join(tokens[answer_start : answer_end + 1])
    else:
        answer = None
        print(
            "I am unable to find the answer to this question. Can you please ask another question?"
        )
    return answer, prof


if __name__ == "__main__":
    data_prep()
    data = pd.read_csv("CoQA_data.csv")
    data.head()

    debug = True
    if debug:
        print("Number of question and answers: ", len(data))

    model = BertForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    for i in range(11):
        print("*" * 20)
        question, text = get_question_text(data)
        answer, prof = get_answer(model, tokenizer, question, text)

        print("\nQuestion:\n{}".format(question.capitalize()))
        if answer is not None:
            print("\nAnswer:\n{}.".format(answer.capitalize()))
        else:
            print("\nAnswer: \nNone")
        print(f"{i} Inference time: {prof}")
