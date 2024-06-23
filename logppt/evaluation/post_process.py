import pandas as pd
import re
from collections import Counter

masking = [
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)(\\/\S\\.[\\S]+)((?=[^A-Za-z0-9])|$)",
     "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})((?=[^A-Za-z0-9])|$)",
     "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)([0-9a-f]{6,} ?){3,}((?=[^A-Za-z0-9])|$)",
     "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)([0-9A-F]{4} ?){4,}((?=[^A-Za-z0-9])|$)",
     "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)(0x[a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)",
     "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)([\\-\\+]?\\d+)((?=[^A-Za-z0-9])|$)", "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)([a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)", "mask_with": "<*>"},
    {"regex_pattern": "(?<=executed cmd )(\".+?\")", "mask_with": "<*>"}
]


def preprocess(line):
    line = line.strip()
    for r in masking:
        line = re.sub(r["regex_pattern"], r["mask_with"], line)
    return " ".join(line.strip().split())


param_regex = [
    r'{([/ :_#.\-\w\d]+)}',
    r'{}'
]

out_dir = "cot_prompt"

datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
            'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']


# datasets = ["HDFS", "Spark", "BGL", "HPC", "Windows", "Linux", "Android", "HealthApp", "Apache", "OpenStack", "Mac"]


def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """
    # template = preprocess(template)

    boolean = {'true', 'false'}
    default_strings = {'null', 'root', 'admin'}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    if user_strings:
        default_strings = default_strings.union(user_strings)

    # apply DS
    if type(template) == float:
        print(template)
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    # apply PS
    p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    new_p_tokens = []
    for p_token in p_tokens:
        if re.match(r'^(\/[^\/]+)+$', p_token):
            p_token = '<*>'
        new_p_tokens.append(p_token)
    template = ''.join(new_p_tokens)

    # tokenize for the remaining rules
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
    new_tokens = []
    for token in tokens:
        # apply BL, US
        for to_replace in boolean.union(default_strings):
            if token.lower() == to_replace.lower():
                token = '<*>'

        # apply DG
        if re.match(r'^\d+$', token):
            token = '<*>'

        # apply WV
        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
                token = '<*>'

        # collect the result
        new_tokens.append(token)

    # make the template using new_tokens
    template = ''.join(new_tokens)

    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    # while "#<*>#" in template:
    #     template = template.replace("#<*>#", "<*>")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>,<*>" in template:
        template = template.replace("<*>,<*>", "<*>")
    return template


if __name__ == '__main__':
    for dname in datasets:
        try:
            log_df = pd.read_csv(f"{out_dir}/{dname}_2k.log_structured.csv")
            content = log_df.Content.tolist()
            template = log_df.EventTemplate.tolist()
            for i in range(len(content)):
                c = content[i]
                t = str(template[i])
                template[i] = correct_single_template(t)
            log_df.EventTemplate = pd.Series(template)

            unique_templates = sorted(Counter(template).items(), key=lambda k: k[1], reverse=True)
            temp_df = pd.DataFrame(unique_templates, columns=['EventTemplate', 'Occurrences'])
            # temp_df.sort_values(by=["Occurrences"], ascending=False, inplace=True)
            log_df.to_csv(f"{out_dir}/{dname}_2k.log_structured_adjusted.csv")
            temp_df.to_csv(f"{out_dir}/{dname}_2k.log_templates_adjusted.csv")
        except Exception as e:
            print(e)
            pass
