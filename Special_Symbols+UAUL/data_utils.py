# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'food general',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']


def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def get_para_asqp_inputs_targets(sents, labels):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    inputs = []
    for i in range(len(labels)):
        cur_sent = sents[i]
        # cur_sent = ' '.join(cur_sent)
        cur_inputs = cur_sent

        # cur_inputs = f"{cur_sent}"
        label = labels[i]
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad

            man_ot = sentword2opinion[sp]

            if at == 'NULL':
                at = 'it'

            one_quad_sentence = f"[AT] {at} [OT] {ot} [AC] {ac} [SP] {man_ot}"
            all_quad_sentences.append(one_quad_sentence)
        target = ' [SSEP] '.join(all_quad_sentences)

        inputs.append(cur_inputs)
        targets.append(target)
    return inputs, targets


def get_transformed_io(data_path, data_dir):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path, silence=False)

    inputs, targets = get_para_asqp_inputs_targets(sents, labels)

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        # './data/rest16/train.txt'
        self.data_path = f'../data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.data_type = data_type
        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "labels": self.all_labels[index]}

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.data_dir)
        self.all_labels = targets
        for i in range(len(inputs)):
            # change input and target to two strings
            # input = ' '.join(inputs[i])
            input = ' '.join(inputs[i])
            target = targets[i]
            if self.data_type == 'train' or self.data_type == 'dev':
                tokenized_input = self.tokenizer.batch_encode_plus(
                  [input], max_length=self.max_len, padding="max_length",
                  truncation=True, return_tensors="pt"
                )
                tokenized_target = self.tokenizer.batch_encode_plus(
                  [target], max_length=self.max_len, padding="max_length",
                  truncation=True, return_tensors="pt"
                )
            else:
                # In order not to destroy the recovery of the label
                data_max_len = 800
                tokenized_input = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.max_len, padding="max_length",
                    truncation=True, return_tensors="pt"
                )
                tokenized_target = self.tokenizer.batch_encode_plus(
                    [target], max_length=data_max_len, padding="max_length",
                    truncation=True, return_tensors="pt"
                )
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
