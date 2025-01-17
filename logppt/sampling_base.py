import textdistance
import random
import time


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
                lengths[i + 1][j +
                               1] = max(lengths[i + 1][j], lengths[i][j + 1])

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
            min_candidate_distance = min(
                min_candidate_distance, jaccard_distance(c_inst, t_inst))
        D.append(min_candidate_distance)
    return D


def adaptive_random_sampling(logs, labels=None, shots=[8], n_candidate=128):
    if labels is None:
        labels = logs.copy()
    # remove duplicates based on logs
    logs, labels = list(zip(*list(set(zip(logs, labels)))))

    sample_results = {}
    sample_set = []
    T = []
    start_time = time.time()
    for shot in range(max(shots)):
        if len(sample_set) == 0:
            i = max(range(0, len(logs)), key=lambda x: (
                len(logs[x].split()), len(logs[x])))
            T.append(logs[i])
            sample_set.append((logs[i], labels[i]))
            del logs[i], labels[i]
            continue
        # candidate_set = [(x, logs[x]) for x in range(len(logs)) if x in random.sample(range(len(logs)), n_candidate)]
        candidate_set = random.sample(list(zip(logs, labels, range(len(logs)))), n_candidate)
        candidate_set = sorted(
            candidate_set, key=lambda x: len(x[0]), reverse=True)
        candidate_distance = min_distance([x[0] for x in candidate_set], T)
        best_candidate = max(range(len(candidate_distance)),
                             key=candidate_distance.__getitem__)
        T.append(candidate_set[best_candidate][0])
        sample_set.append((candidate_set[best_candidate][0], candidate_set[best_candidate][1]))
        if shot + 1 in shots:
            sample_results[shot + 1] = sample_set.copy()
            print(f"{shot + 1}-shot sampling time: ", (time.time() - start_time))
            start_time = time.time()
        del logs[candidate_set[best_candidate][2]], labels[candidate_set[best_candidate][2]]
    return sample_results
