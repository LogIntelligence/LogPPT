import re
from transformers import DataCollatorForTokenClassification
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_logformat_regex(logformat):
    """ 
    Function to generate regular expression to split log messages
    Args:
        logformat: log format, a string
    Returns:
        headers: headers of log messages
        regex: regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
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


def log_to_dataframe(log_file, regex, headers, size=None):
    """ 
    Function to transform log file to contents
    Args:
        log_file: log file path
        regex: regular expression to split log messages
        headers: headers of log messages
        size: number of log messages to read
    Returns:
        log_messages: list of log contents
    """
    log_contents = []
    with open(log_file, 'r') as file:
        if size is None:
            log_lines = file.readlines()
        else:
            log_lines = [next(file) for _ in range(size)]
        for line in log_lines:
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_contents.append(message[-1])
            except Exception as e:
                pass
    return log_contents


def get_initial_label_words(model, loader, val_label=1):
    initial_label_words = []
    for batch in loader:
        label = batch.pop('ori_labels')
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'])
        logits = torch.topk(outputs.logits.log_softmax(dim=-1), k=5).indices
        for i in range(label.shape[0]):
            for j in range(len(label[i])):
                if label[i][j] != val_label:
                    continue
                initial_label_words.extend(
                    logits[i][j].detach().cpu().clone().tolist())
                # initial_label_words.append(int(batch['input_ids'][i][j]))

    return list(set(initial_label_words))


def find_label_words(model, train_loader, eval_loader, val_label=1, no_label_words=8):
    model.to(device)
    model.eval()
    initial_label_words = get_initial_label_words(
        model, train_loader, val_label)
    label_words_freq = {k: 0 for k in initial_label_words}
    for batch in eval_loader:
        for (inp, label) in zip(batch['input_ids'].detach().clone().tolist(), batch['ori_labels'].detach().clone().tolist()):
            for i, l in zip(inp, label):
                if l != val_label:
                    continue
                try:
                    label_words_freq[i] += 1
                except KeyError:
                    pass
    label_words_freq = {x[0]: x[1] for x in sorted(
        label_words_freq.items(), key=lambda k: k[1], reverse=True)}
    label_words = list(label_words_freq.keys())[:no_label_words]
    return label_words


class LogParsingDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        ori_labels = [feature['ori_labels'] for feature in features] if 'ori_labels' in features[0].keys() else None
        max_length = max([len(x['input_ids']) for x in features])
        # print(max_length)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=min(max_length, 256),
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch
        # print(batch['labels'])
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            batch['ori_labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in
                                   ori_labels]
            # batch['ori_labels'] = None
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            batch["ori_labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in
                                   ori_labels]
            # batch['ori_labels'] = None
        # print(batch)
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch
    
