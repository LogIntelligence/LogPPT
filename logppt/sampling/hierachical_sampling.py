import json
import os
import pandas as pd
import re
import string
from sklearn.utils import shuffle
import textdistance
import random
import heapq
from collections import Counter, defaultdict, deque, OrderedDict
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import time
import calendar
import argparse
import numpy as np
from copy import deepcopy

# from . import datasets, benchmark

def generate_logformat_regex(log_format):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', log_format)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, log_format):
    """ Function to transform log file to dataframe
    """
    headers, regex = generate_logformat_regex(log_format)
    log_messages = []
    line_count = 0
    with open(log_file, 'r', encoding='utf8', errors='ignore') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                line_count += 1
            except Exception as _:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(line_count)]
    return logdf


def lcs_distance(x, y):
    seq1 = x.split()
    seq2 = y.split()
    lengths = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    return 1 - 2 * lengths[-1][-1] / (len(seq1) + len(seq2))


def lev_distance(x, y):
    return textdistance.levenshtein.normalized_distance(x, y)


def euc_distance(x, y):
    return textdistance.cosine.normalized_distance(x, y)


def jaccard_distance(x, y):
    return textdistance.jaccard.normalized_distance(x.split(), y.split())


def ratcliff_distance(x, y):
    return textdistance.ratcliff_obershelp.normalized_distance(x, y)


def min_distance(c_set, t_set):
    D = []
    for c_inst in c_set:
        min_candidate_distance = 1e10
        for t_inst in t_set:
            min_candidate_distance = min(min_candidate_distance, jaccard_distance(c_inst, t_inst))
        D.append(min_candidate_distance)
    return D


def adaptive_random_sampling(logs, labels=None, shot=8):
    if shot >= len(logs):
        return list(zip(logs, labels))
    if labels is None:
        labels = logs.copy()
    sample_set = []
    T = []
    while shot > 0:
        if len(sample_set) == 0:
            i = max(range(0, len(logs)), key=lambda x: (
                len(logs[x].split()), len(logs[x])))
            T.append(logs[i])
            sample_set.append((logs[i], labels[i]))
            del logs[i], labels[i]
            shot -= 1
            continue
        n_candidate = min(8, len(logs))
        candidate_set = random.sample(list(zip(logs, labels, range(len(logs)))), n_candidate)
        candidate_set = sorted(
            candidate_set, key=lambda x: len(x[0]), reverse=True)
        candidate_distance = min_distance([x[0] for x in candidate_set], T)
        best_candidate = max(range(len(candidate_distance)),
                             key=candidate_distance.__getitem__)
        T.append(candidate_set[best_candidate][0])
        sample_set.append((candidate_set[best_candidate][0], candidate_set[best_candidate][1]))
        del logs[candidate_set[best_candidate][2]], labels[candidate_set[best_candidate][2]]
        shot -= 1
    return sample_set


class Vocab:
    def __init__(self, stopwords=["<*>"]):
        stopwords = [
            "a",
            "an",
            "and",
            "i",
            "ie",
            "so",
            "to",
            "the",

        ] + list(calendar.day_name) + list(calendar.day_abbr) \
          + list(calendar.month_name) + list(calendar.month_abbr)
        self.token_counter = Counter()
        self.stopwords = frozenset(set(stopwords))
        #print(self.__filter_stopwords(['LDAP', 'Built', 'with']))

    def build(self, sequences):
        print("Build vocab with examples: ", len(sequences))
        for sequence in sequences:
            sequence = self.__filter_stopwords(sequence)
            #print(sequence)
            self.update(sequence)

    def update(self, sequence):
        sequence = self.__filter_stopwords(sequence)
        self.token_counter.update(sequence)

    def topk_tokens(self, sequence, topk=3):
        sequence = self.__filter_stopwords(sequence)
        token_count = [(token, self.token_counter[token]) for token in set(sequence)]
        topk_tuples = heapq.nlargest(topk, token_count, key=lambda x: x[1])
        topk_keys = tuple([t[0] for t in topk_tuples])
        return topk_keys

    def __len__(self):
        return len(self.token_counter)

    def __filter_stopwords(self, sequence):
        return [
            token
            for token in sequence
            if (len(token) > 2) and (token not in self.stopwords)
        ]


def clean(s):
    log_format = re.sub(r'[0-9A-Za-z, ]+', '', s)
    unique_chars = list(set(log_format))
    sorted_string = ''.join(sorted(unique_chars))
    s = re.sub(':|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.?!', ' ', s)
    s = " ".join([word for word in s.strip().split() if not bool(re.search(r'\d', word))])
    # trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    return s, sorted_string


def hierarchical_clustering(contents):
    vocab = Vocab()
    vocab.build([v[0].split() for v in contents.values()])

    # hierarchical clustering
    hierarchical_clusters = {}
    for k, v in contents.items():
        frequent_token = tuple(sorted(vocab.topk_tokens(v[0].split(), 3))) 
        log_format = v[1]
        if frequent_token not in hierarchical_clusters:
            hierarchical_clusters[frequent_token] = {"size": 1, "cluster": {log_format: [k]}}
        else:
            hierarchical_clusters[frequent_token]["size"] = hierarchical_clusters[frequent_token]["size"] + 1
            if log_format not in hierarchical_clusters[frequent_token]["cluster"]:
                hierarchical_clusters[frequent_token]["cluster"][log_format] = [k]
            else:
                hierarchical_clusters[frequent_token]["cluster"][log_format].append(k)
    print("Number of coarse-grained clusters: ", len(hierarchical_clusters.keys()))
    total_fine_clusters = 0
    for k, v in hierarchical_clusters.items():
        total_fine_clusters += len(hierarchical_clusters[k]["cluster"])
    print("Number of fine-grained clusters: ", total_fine_clusters)
    return hierarchical_clusters


def hierarchical_distribute(hierarchical_clusters, shot, logs=[], labels=[]):
    candidate_samples = []
    coarse_clusters = hierarchical_clusters.keys()
    coarse_clusters = shuffle(list(coarse_clusters))
    corase_size = len(coarse_clusters)
    coarse_quotas = [0] * corase_size
    while shot > 0:
        round_quota = 0
        for coarse_id, coarse_key in enumerate(coarse_clusters):
            if coarse_quotas[coarse_id] == hierarchical_clusters[coarse_key]["size"]:
                continue
            coarse_quota = min(int(shot // corase_size) + (coarse_id < shot % corase_size), hierarchical_clusters[coarse_key]["size"] - coarse_quotas[coarse_id])
            if coarse_quota == 0:
                coarse_quota = 1
            coarse_quotas[coarse_id] += coarse_quota
            round_quota += coarse_quota
            if round_quota == shot:
                break
        shot -= round_quota
    for coarse_id, coarse_key in enumerate(coarse_clusters):
        # coarse_quota = min(int(shot // corase_size) + (coarse_id < shot % corase_size), hierarchical_clusters[coarse_key]["size"])
        # if coarse_quota == 0:
        #     break
        coarse_quota = coarse_quotas[coarse_id]
        fine_clusters = hierarchical_clusters[coarse_key]["cluster"].keys()
        fine_clusters = sorted(fine_clusters, key=lambda x: len(hierarchical_clusters[coarse_key]["cluster"][x]), reverse=True)
        fine_size = len(fine_clusters)
        fine_quotas = [0] * fine_size
        while coarse_quota > 0:
            round_quota = 0
            for fine_id, fine_key in enumerate(fine_clusters):
                if fine_quotas[fine_id] == len(hierarchical_clusters[coarse_key]["cluster"][fine_key]):
                    continue
                fine_quota = min(int(coarse_quota // fine_size) + (fine_id < coarse_quota % fine_size), len(hierarchical_clusters[coarse_key]["cluster"][fine_key]) - fine_quotas[fine_id])
                if fine_quota == 0:
                    fine_quota = 1
                fine_quotas[fine_id] += fine_quota
                round_quota += fine_quota
                if round_quota == coarse_quota:
                    break
            coarse_quota -= round_quota

        print("Fine quotas: ", fine_quotas)
        # assert sum(fine_quotas) == shot, "Quota mismatch"

        for fine_id, fine_key in enumerate(fine_clusters):
            # fine_quota = int(coarse_quota // fine_size) + (fine_id < coarse_quota % fine_size)
            fine_quota = fine_quotas[fine_id]
            if fine_quota == 0:
                break
        
            cluster_ids = hierarchical_clusters[coarse_key]["cluster"][fine_key]
            cluster_logs = [logs[i] for i in cluster_ids]
            cluster_labels = [labels[i] for i in cluster_ids]
            
            assert fine_quota <= len(cluster_logs), "Quota mismatch"
            # samples = adaptive_random_sampling(cluster_logs, cluster_labels, fine_quota)
            # randomly sample from the cluster
            samples = random.sample(list(zip(cluster_logs, cluster_labels)), fine_quota)
            candidate_samples.extend(samples)

    return candidate_samples


def sampling(logs, labels=None, shots=[8]):
    # only keep unique logs with the corresponding labels
    logs, labels = zip(*list(set(zip(logs, labels))))
    contents = {}
    for i, x in enumerate(logs):
        x, fx = clean(x)
        if len(x.split()) > 0:
            contents[i] = (x, fx)
    # content = {i: clean(x) if len(x.split()) > 1 for i, x in enumerate(labelled_logs['Content'].tolist())}
    begin_time = time.time()
    hierarchical_clusters = hierarchical_clustering(contents)
    end_time = time.time()
    clustering_time = end_time - begin_time
    print("hierarchical clustering time: ", clustering_time)
    sample_candidates = {}
    for idx, shot in enumerate(shots):
        begin_time = time.time()
        samples = hierarchical_distribute(deepcopy(hierarchical_clusters), shot, logs, labels)
        # if labels is not None:
        #     samples = [(logs[i], labels[i]) for i in sampled_ids]
        # else:
        #     samples = [(logs[i], logs[i]) for i in sampled_ids]
        sample_candidates[shot] = samples
        end_time = time.time()
        print(f"{shot}-shot sampling time: ", (end_time - begin_time))

    return sample_candidates



def hierarchical_distribute2(hierarchical_clusters, shot, logs=[], labels=[]):
    candidate_samples = []
    coarse_clusters = hierarchical_clusters.keys()
    # coarse_clusters = shuffle(list(coarse_clusters))
    coarse_clusters = sorted(coarse_clusters, key=lambda x: hierarchical_clusters[x]["size"], reverse=True)
    corase_size = len(coarse_clusters)
    coarse_quotas = [0] * corase_size
    while shot > 0:
        round_quota = 0
        for coarse_id, coarse_key in enumerate(coarse_clusters):
            if coarse_quotas[coarse_id] == hierarchical_clusters[coarse_key]["size"]:
                continue
            coarse_quota = min(int(shot // corase_size) + (coarse_id < shot % corase_size), hierarchical_clusters[coarse_key]["size"] - coarse_quotas[coarse_id])
            if coarse_quota == 0:
                coarse_quota = 1
            coarse_quotas[coarse_id] += coarse_quota
            round_quota += coarse_quota
            if round_quota == shot:
                break
        shot -= round_quota
    for coarse_id, coarse_key in enumerate(coarse_clusters):
        coarse_quota = coarse_quotas[coarse_id]
        logs_ids = []
        for _, log_ids in hierarchical_clusters[coarse_key]["cluster"].items():
            logs_ids.extend(log_ids)
        logs = [logs[i] for i in logs_ids]
        labels = [labels[i] for i in logs_ids]
        samples = adaptive_random_sampling(logs, labels, coarse_quota)
        candidate_samples.extend(samples)

    return candidate_samples