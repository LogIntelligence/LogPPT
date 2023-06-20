import json
import os
import pandas as pd
import re
import string
from sklearn.utils import shuffle

from logppt.sampling import adaptive_random_sampling
from logppt.utils import log_to_dataframe, benchmark


def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message
    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    s = re.sub(':|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    s = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in s.strip().split()])
    return s


if __name__ == '__main__':
    os.makedirs("datasets", exist_ok=True)
    for dataset in benchmark.keys():
        print(dataset)
        setting = benchmark[dataset]
        os.makedirs("datasets/{0}".format(dataset), exist_ok=True)

        logdf = log_to_dataframe(f'./logs/{setting["log_file"]}', setting['log_format'])
        logdf.to_csv(f"datasets/{setting['log_file']}_structured.csv")
        content = [(clean(x), i, len(x)) for i, x in enumerate(logdf['Content'].tolist())]
        content = [x for x in content if len(x[0].split()) > 1]
        content = shuffle(content)

        for shot in [4, 8, 16, 32]:
            keywords_list = []
            os.makedirs("datasets/{0}/{1}shot".format(dataset, shot), exist_ok=True)
            samples_ids = adaptive_random_sampling(shuffle(content), shot)

            labeled_samples = [(row['Content'], row['Content']) for _, row in logdf.take(samples_ids).iterrows()]
            labeled_samples = [{"text": x[0], "label": x[1], "type": 1} for x in labeled_samples]
            with open("datasets/{0}/{1}shot/{2}.json".format(dataset, shot, 1), "w") as f:
                for s in labeled_samples:
                    f.write(json.dumps(s) + "\n")
