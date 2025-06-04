import ast
import astor
import io
import tokenize
import keyword
import json
from io import StringIO
from collections import defaultdict
from reindent import run as run_reindent

def normalize_node(node):
    if isinstance(node, ast.Name):
        return "Name"
    elif isinstance(node, ast.Constant):
        return "Const"
    else:
        children = [normalize_node(child) for child in ast.iter_child_nodes(node)]
        return "{}({})".format(type(node).__name__, ",".join(children))

def get_normalized_nodes_by_line(tree):
    line_dict = {}
    for node in ast.walk(tree):
        if hasattr(node, 'lineno'):
            rep = normalize_node(node)
            line_dict.setdefault(node.lineno, set()).add(rep)
    return line_dict

def get_common_tokens1(codes):
    common = None
    for code in codes:
        tokens = set()
        reader = io.StringIO(code)
        for tok in tokenize.generate_tokens(reader.readline):
            if tok.type not in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}:
                tokens.add(tok.string)
        if common is None:
            common = tokens
        else:
            common = common.intersection(tokens)
    return common

THRESHOLD = int

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

def get_common_tokens(efficient_solutions):
    min_count = (len(efficient_solutions) + 1) // 2
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

def generate_skeleton_code_with_common_lines(code, line_time, common_lines, common_tokens, threshold=THRESHOLD):
    lines = code.splitlines()
    output_lines = [""] * len(lines)
    reader = io.StringIO(code)
    tokens = list(tokenize.generate_tokens(reader.readline))
    
    line_tokens = {}
    for tok in tokens:
        tok_type = tok.type
        tok_string = tok.string
        start_line, start_col = tok.start
        if tok_type in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}:
            continue
        if start_line in common_lines and line_time.get(start_line, 0) >= threshold:
            if tok_type == tokenize.OP or tok_string in keyword.kwlist or tok_string in common_tokens:
                keep_str = tok_string
            else:
                keep_str = ""
            line_tokens.setdefault(start_line, []).append((start_col, keep_str))
    
    for i in range(1, len(lines)+1):
        if i in line_tokens:
            tokens_in_line = sorted(line_tokens[i], key=lambda x: x[0])
            line_result = ""
            current_col = 0
            for col, s in tokens_in_line:
                if col > current_col:
                    line_result += "<MASK>" * (col - current_col)
                line_result += s
                current_col = col + len(s)
            output_lines[i-1] = line_result
        else:
            output_lines[i-1] = ""
    return "\n".join(output_lines)

with open('', 'r', encoding='utf-8') as file:
    efficientCode = json.load(file)
with open('', 'r', encoding='utf-8') as file:
    efficientCodeLineTime = json.load(file)
efficientCodeSkeleton_dict = {}
for i in range (0, 5000):
    num = str(i).zfill(4)
    if num in efficientCode:
        data_num = efficientCode[num]
        line_time_list = efficientCodeLineTime[num]
        codes = data_num["codes"]
        efficientCodeSkeletons = []
        efficientCodeCommon = []
        common = []
        try:
            efficientCodeCommon = get_common_tokens(codes)
        except Exception as e:
            print(f"{e},key:{num}")
        for j in range (0, 5):
            normalized_sets = []
            for code in codes:
                tree = ast.parse(code)
                norm_nodes = set()
                for node in ast.walk(tree):
                    if hasattr(node, 'lineno'):
                        rep = normalize_node(node)
                        norm_nodes.add(rep)
                normalized_sets.append(norm_nodes)

            if normalized_sets:
                global_common = set.intersection(*normalized_sets)
            else:
                global_common = set()
            sample2_tree = ast.parse(codes[j])
            sample2_line_nodes = get_normalized_nodes_by_line(sample2_tree)
            common_lines_improved = {line for line, reps in sample2_line_nodes.items() if reps.intersection(global_common)}
            common_tokens = get_common_tokens1(codes)
            line_time = line_time_list[j]
            line_time = {int(k): v for k, v in line_time.items()}
            try:
                skeleton_code_improved = generate_skeleton_code_with_common_lines(codes[j], line_time, common_lines_improved, common_tokens, 0.1)
                original_list = get_tokens_from_code(skeleton_code_improved)
                unique_list = []
                [unique_list.append(item) for item in original_list if item not in unique_list]
            except Exception as e:  
                print(f"{num,j},{e}")
            common = efficientCodeCommon + unique_list
            c = reindent_code(codes[j])
            efficientCodeSkeleton = filter_code(c,common)
            efficientCodeSkeletons.append(efficientCodeSkeleton)
        efficientCodeSkeleton_dict[num] = efficientCodeSkeletons

filename = ""

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(efficientCodeSkeleton_dict, f, indent=4)  
