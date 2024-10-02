import regex as re
import string

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
            # if token.lower() == to_replace.lower():
            if token == to_replace.lower():
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
    #print("CV: ", template)
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        template = re.sub(r'<\*>\:<\*>', '<*>', template)
        template = re.sub(r'<\*> <\*>', '<*>', template)
        if prev == template:
            break
    # while "<*>:<*>" in template:
    #     template = template.replace("<*>:<*>", "<*>")

    return template


# import re
# import string

# def correct_single_template(template, user_strings=None):
#     """Apply all rules to process a template.

#     DS (Double Space)
#     BL (Boolean)
#     US (User String)
#     DG (Digit)
#     PS (Path-like String)
#     WV (Word concatenated with Variable)
#     DV (Dot-separated Variables)
#     CV (Consecutive Variables)

#     """

#     boolean = {}
#     default_strings = {}
#     path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
#         r'\s', r'\,', r'\!', r'\;', r'\:',
#         r'\=', r'\|', r'\"', r'\'',
#         r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
#     }
#     token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
#         r'\+', r'\@', r'\#',
#     })

#     if user_strings:
#         default_strings = default_strings.union(user_strings)

#     # apply DS
#     template = template.strip()
#     template = re.sub(r'\s+', ' ', template)

#     # apply PS
#     # p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
#     # new_p_tokens = []
#     # for p_token in p_tokens:
#     #     # if re.match(r'^(\/[^\/]+)+$', p_token):
#     #     #     p_token = '<*>'
#     #     new_p_tokens.append(p_token)
#     # template = ''.join(new_p_tokens)

#     # tokenize for the remaining rules
#     tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
#     new_tokens = []
#     # print(tokens)
#     for token in tokens:
#         # apply BL, US
#         # for to_replace in boolean.union(default_strings):
#         #     if token.lower() == to_replace.lower():
#         #         token = '<*>'

#         # apply DG
#         if re.match(r'^\d+$', token):
#             token = '<*>'

#         # apply WV
#         if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
#             if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
#                 token = '<*>'

#         # collect the result
#         new_tokens.append(token)

#     # make the template using new_tokens
#     template = ''.join(new_tokens)

#     # Substitute consecutive variables only if separated with any delimiter including "." (DV)
#     while True:
#         prev = template
#         template = re.sub(r'<\*>\.<\*>', '<*>', template)
#         if prev == template:
#             break


#     while " #<*># " in template:
#         template = template.replace(" #<*># ", " <*> ")

#     while " #<*> " in template:
#         template = template.replace(" #<*> ", " <*> ")

#     while "<*>:<*>" in template:
#         template = template.replace("<*>:<*>", "<*>")

#     while "<*>#<*>" in template:
#         template = template.replace("<*>#<*>", "<*>")

#     while "<*>/<*>" in template:
#         template = template.replace("<*>/<*>", "<*>")

#     while "<*>@<*>" in template:
#         template = template.replace("<*>@<*>", "<*>")

#     while "<*>.<*>" in template:
#         template = template.replace("<*>.<*>", "<*>")

#     # while "<*>,<*>" in template:
#     #     template = template.replace("<*>,<*>", "<*>")

#     while ' "<*>" ' in template:
#         template = template.replace(' "<*>" ', ' <*> ')

#     while " '<*>' " in template:
#         template = template.replace(" '<*>' ", " <*> ")

#     while "<*><*>" in template:
#         template = template.replace("<*><*>", "<*>")

#     # while "[<*>]" in template:
#     #     template = template.replace("[<*>]", "<*>")

#     # while "(<*>)" in template:
#     #     template = template.replace("(<*>)", "<*>")

#     # while "{<*>}" in template:
#     #     template = template.replace("{<*>}", "<*>")    

#     # while "<*> <*>" in template:
#     #     template = template.replace("<*> <*>", "<*>")
#     return template

#     # def verify_template(token):
#     #     token = token.replace("<*>", "")
#     #     token = token.replace(" ", "")
#     #     return any(char not in string.punctuation for char in template)
    
#     # tokens = template.split(" ")
#     # tokens = [token for token in tokens if verify_template(token)]
    
#     # return " ".join(tokens)



def correct_single_template_for_evaluation(template, user_strings=None):
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