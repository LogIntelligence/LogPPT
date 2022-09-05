import json
import os
import pandas as pd
import re
import string
from logflap.sampling import adaptive_random_sampling
from sklearn.utils import shuffle


def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message
    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    s = re.sub(':|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    s = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in s.strip().split()])
    return s


if __name__ == '__main__':

    for dataset in ["Android", "Apache", "BGL", "Hadoop", "HDFS", "HealthApp", "HPC", "Linux", "Mac", "OpenSSH",
                    "OpenStack", "Proxifier", "Spark", "Thunderbird", "Windows", "Zookeeper"]:
        # for dataset in ["Thunderbird"]:
        print(dataset)
        log = pd.read_csv("./logs/{0}/{1}_2k.log_structured_corrected.csv".format(dataset, dataset))
        train_df = log.sample(n=2000)
        samples = [(row['Content'], row['EventTemplate']) for _, row in log.iterrows()]
        # print(samples)
        # samples = [gen_input_label(x[0], x[1], []) for x in samples]
        samples = [{"text": x[0], "label": x[1], "type": 1} for x in samples]
        os.makedirs("datasets/log_parsing/{0}".format(dataset), exist_ok=True)
        with open("datasets/log_parsing/{0}/test.json".format(dataset), "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        content = [(clean(x), i, len(x)) for i, x in enumerate(log['Content'].tolist())]
        content = [x for x in content if len(x[0].split()) > 1]

        for shot in [4, 8, 16, 32]:
            keywords_list = []
            os.makedirs("datasets/log_parsing/{0}/{1}shot".format(dataset, shot), exist_ok=True)
            samples_ids = adaptive_random_sampling(shuffle(content), shot)
            print(shot, samples_ids)
            labeled_samples = [(row['Content'], row['EventTemplate']) for _, row in log.take(samples_ids).iterrows()]
            labeled_samples = [{"text": x[0], "label": x[1], "type": 1} for x in labeled_samples]
            with open("datasets/log_parsing/{0}/{1}shot/{2}.json".format(dataset, shot, 1), "w") as f:
                for s in labeled_samples:
                    f.write(json.dumps(s) + "\n")
