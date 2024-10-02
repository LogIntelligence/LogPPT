import re
from torch.utils.data import DataLoader

from logppt.data.base import BaseDataLoader
from logppt.data.utils import LogParsingDataCollator, find_label_words

delimiters = "([ |\(|\)|\[|\]|\{|\})])"

def align_with_null_values(log, template):
    """
    Align the null values
    """

    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  
    matches = re.search(regex, log)

    if matches == None:
        return template

    parts = []
    for index, part in enumerate(template.split("<*>")):
        parts.append(part)
        if index < len(matches.groups()):
            if matches.groups()[index] == '':
                parts.append('')
            else:
                parts.append('<*>')
    return ''.join(parts)


def get_template_regex(template):
    """
    :param template: template regex with <*> indicates parameters
    :return: regex pattern
    """
    # template_regex = re.sub(r"<.{1,5}>", "<*>", template_regex)
    if "<*>" not in template:
        return None
    template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template)
    template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
    template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
    return template_regex

class DataLoaderForPromptTuning(BaseDataLoader):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.load_data()

    def initialize(self, tokenizer):
        self.tokenizer = tokenizer

        self.tokenizer.add_tokens(self.vtoken)

        self.label_to_id = {
            'o': 0,
            self.vtoken: 1
        }
        self.vtoken_id = self.tokenizer.convert_tokens_to_ids(self.vtoken)

        return self.tokenizer
    
    def tokenize(self):
        def tokenize_and_align_labels(examples):
            target_tokens = []
            tag_labels = []
            input_ids = []
            attention_masks = []
            for i, (log, label) in enumerate(zip(examples[self.text_column_name], examples[self.label_column_name])):
                log = " ".join(log.strip().split())
                label = " ".join(label.strip().split())
                template_regex = get_template_regex(label)
                if template_regex is None:
                    input_tokens, label_tokens = [log], ['o']
                else:
                    match = next(re.finditer(template_regex, log))
                    input_tokens = []
                    label_tokens = []
                    cur_position = 0
                    for idx in range(match.lastindex):
                        start, end = match.span(idx + 1)
                        if start > cur_position:
                            input_tokens.append(log[cur_position:start].rstrip())
                            label_tokens.append("o")
                        input_tokens.append(log[start:end])
                        if start > 0 and log[start - 1] == " ":
                            input_tokens[-1] = " " + input_tokens[-1]
                        label_tokens.append(self.vtoken)
                        cur_position = end
                        
                    if cur_position < len(log):
                        input_tokens.append(log[cur_position:len(log)].rstrip())
                        label_tokens.append('o')
                        
                refined_tokens = []
                refined_labels = []
                for (t, l) in zip(input_tokens, label_tokens):
                    if len(t) == 0:
                        continue
                    t = re.split(delimiters, t)
                    t = [x for x in t if len(x) > 0]
                    sub_tokens = []
                    if t[0] != " ":
                        sub_tokens.append(t[0])
                    for i in range(1, len(t)):
                        if t[i] == " ":
                            continue
                        if t[i - 1] == " ":
                            sub_tokens.append(" " + t[i])
                        else:
                            sub_tokens.append(t[i])
                    refined_tokens.extend(sub_tokens)
                    refined_labels.extend([l] * len(sub_tokens))
                    
                input_id = []
                labels = []
                target_token = []
                # print(i, input_tokens, label_tokens)
                for (input_token, label_token) in zip(refined_tokens, refined_labels):
                    token_ids = self.tokenizer.encode(
                        input_token, add_special_tokens=False)
                    input_id.extend(token_ids)
                    if label_token != self.vtoken:
                        target_token.extend(token_ids)
                    else:
                        target_token.extend([self.vtoken_id] * len(token_ids))
                        # print(input_token)
                    labels.extend([self.label_to_id[label_token]]
                                  * len(token_ids))
                input_id = [self.tokenizer.cls_token_id] + \
                    input_id + [self.tokenizer.sep_token_id]
                target_token = [self.tokenizer.bos_token_id] + target_token + [self.tokenizer.eos_token_id]
                labels = [-100] + labels + [-100]
                attention_mask = [1] * len(input_id)
                input_ids.append(input_id)
                target_tokens.append(target_token)
                tag_labels.append(labels)
                attention_masks.append(attention_mask)
            return {
                "input_ids": input_ids,
                "labels": target_tokens,
                "ori_labels": tag_labels,
                "attention_mask": attention_masks
            }

        self.processed_raw_datasets = {}
        self.processed_raw_datasets['train'] = self.raw_datasets['train'].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=self.raw_datasets["train"].column_names,
            desc="Running tokenizer on train dataset",
            num_proc=1
        )
        if 'validation' in self.raw_datasets:
            self.processed_raw_datasets['validation'] = self.raw_datasets['validation'].map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=self.raw_datasets["validation"].column_names,
                desc="Running tokenizer on test dataset",
                num_proc=4
            )
    
    def build_dataloaders(self, per_device_train_batch_size, per_device_eval_batch_size):
        data_collator = LogParsingDataCollator(
            tokenizer=self.tokenizer, pad_to_multiple_of=None)
        self.train_loader = DataLoader(
            self.processed_raw_datasets['train'],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size
        )
        if 'validation' in self.processed_raw_datasets:
            self.val_loader = DataLoader(
                self.processed_raw_datasets['validation'],
                collate_fn=data_collator,
                batch_size=per_device_eval_batch_size
            )
        else:
            self.val_loader = None

    def find_parameter_label_words(self, model, no_label_words=1):
        return {
            self.vtoken: find_label_words(model, self.train_loader, self.val_loader, no_label_words=no_label_words)
        }