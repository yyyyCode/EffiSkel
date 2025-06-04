import os
import json
import token_common as tc
from reindent import run as run_reindent
import io
import tokenize
from io import StringIO
from functools import reduce

def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )

    return ret.getvalue()
# 从代码字符串中提取token
def get_tokens_from_code(code):
    tokens = []
    code_io = StringIO(code)
    for token in tokenize.generate_tokens(code_io.readline):
        token_type = token.type
        token_string = token.string
        if token_type != tokenize.COMMENT and token_string.strip():
            tokens.append(token_string)
    return tokens

def get_common_tokens(codes):
    token_lists = [set(get_tokens_from_code(code)) for code in codes]
    common_tokens = reduce(lambda a, b: a & b, token_lists)  
    return sorted(common_tokens)

def filter_code(code, valid_tokens):
    filtered_code = []
    
    for line in code.split('\n'):
        new_line = []
        token = ''
        
        for char in line:
            if char.isalnum() or char in "_":  
                token += char
            else:
                if token:
                    if token in valid_tokens:
                        new_line.append(token) 
                    else:
                        new_line.append('<MASK>') 
                    token = ''  
                new_line.append(char) 
        
        if token:
            if token in valid_tokens:
                new_line.append(token)
            else:
                new_line.append(' ' * len(token))
        
        filtered_code.append(''.join(new_line)) 
    
    return '\n'.join(filtered_code)

efficientCode_dict_json = ""
efficientCodeSkeleton_dict = {}
with open(efficientCode_dict_json,'r',encoding='utf-8') as e:
    dict_data = json.load(e)
for key, values in dict_data.items():
    efficientCode = []
    efficientCodeSkeletons = []
    value = values["codes"]
    for i in value:
        code = i
        efficientCode.append(code)
    efficientCodeCommon = get_common_tokens(efficientCode)
    for j in efficientCode:
        j = reindent_code(j)
        efficientCodeSkeleton = filter_code(j,efficientCodeCommon)
        efficientCodeSkeletons.append(efficientCodeSkeleton)
    efficientCodeSkeleton_dict[key] = efficientCodeSkeletons

filename = ""

with open(filename, 'w') as f:
    json.dump(efficientCodeSkeleton_dict, f, indent=2)
