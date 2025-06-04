import json
from reindent import run as run_reindent
import io
import tokenize
from io import StringIO
from functools import reduce
import ast
import astor 
from collections import defaultdict

def code_to_ast_with_main_node(code):
    tree = ast.parse(code)
    if not any(isinstance(node, ast.FunctionDef) for node in tree.body):
        main_node = ast.FunctionDef(
            name="main",
            args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=tree.body,
            decorator_list=[]
        )
        tree = ast.Module(body=[main_node], type_ignores=[])
    return tree

def extract_subtrees(node):
    subtrees = []
    for child in ast.iter_child_nodes(node):
        subtrees.append(child)
        subtrees.extend(extract_subtrees(child))
    return subtrees

def are_trees_similar(tree1, tree2, tolerance=float):
    def get_node_structure(node):
        return [type(n).__name__ for n in ast.walk(node)]
    
    struct1 = get_node_structure(tree1)
    struct2 = get_node_structure(tree2)
    
    common_nodes = set(struct1).intersection(struct2)
    similarity = len(common_nodes) / max(len(struct1), len(struct2))
    return similarity >= tolerance

def find_frequent_subtrees(solutions, min_count):
    ast_trees = [code_to_ast_with_main_node(code) for code in solutions]
    all_subtrees = [extract_subtrees(tree) for tree in ast_trees]
    
    subtree_frequency = defaultdict(int)
    subtree_objects = {}

    for i in range(len(all_subtrees)):
        for subtree1 in all_subtrees[i]:
            subtree1_dump = ast.dump(subtree1)
            if subtree1_dump in subtree_objects:
                continue
            for j in range(i + 1, len(all_subtrees)):
                for subtree2 in all_subtrees[j]:
                    if are_trees_similar(subtree1, subtree2):
                        subtree_frequency[subtree1_dump] += 1
                        subtree_objects[subtree1_dump] = subtree1
                        break

    frequent_subtrees = [subtree_objects[dump] for dump, count in subtree_frequency.items() if count + 1 >= min_count]
    return frequent_subtrees

def generate_skeleton(subtrees):
    skeleton_code = ""
    for subtree in subtrees:
        try:
            skeleton_code += astor.to_source(subtree)
        except Exception as e:
            continue
    return skeleton_code

def get_tokens_from_code(code):
    tokens = []
    code_io = StringIO(code)
    for token in tokenize.generate_tokens(code_io.readline):
        token_type = token.type
        token_string = token.string
        if token_type != tokenize.COMMENT and token_string.strip():
            tokens.append(token_string)
    return tokens

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
def get_tokens_from_code(code):
    tokens = []
    code_io = StringIO(code)
    for token in tokenize.generate_tokens(code_io.readline):
        token_type = token.type
        token_string = token.string
        if token_type != tokenize.COMMENT and token_string.strip():
            tokens.append(token_string)
    return tokens

def get_common_tokens(efficient_solutions):
    min_count = int
    frequent_subtrees = find_frequent_subtrees(efficient_solutions, min_count=min_count)
    skeleton = generate_skeleton(frequent_subtrees)
    original_list = get_tokens_from_code(skeleton)
    unique_list = []
    [unique_list.append(item) for item in original_list if item not in unique_list]
    return unique_list

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
for key, values in  dict_data.items():
    efficientCode = []
    efficientCodeSkeletons = []
    value = values["codes"]
    for i in value:
        code = i
        efficientCode.append(code)
    try:
        efficientCodeCommon = get_common_tokens(efficientCode)
    except Exception as e:
        print(f"{e},key:{key}")
    for j in efficientCode:
        j = reindent_code(j)
        efficientCodeSkeleton = filter_code(j,efficientCodeCommon)
        efficientCodeSkeletons.append(efficientCodeSkeleton)
    efficientCodeSkeleton_dict[key] = efficientCodeSkeletons

filename = ""

with open(filename, 'w') as f:
    json.dump(efficientCodeSkeleton_dict, f, indent=2) 