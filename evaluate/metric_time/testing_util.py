import json
import io
import sys
import os
import re
from typing import List
from datetime import datetime
from enum import Enum

# to run the solution files we're using a timing based approach
import signal
from pyext import RuntimeModule
from tqdm import tqdm
import gc
import faulthandler
# for capturing the stdout
from io import StringIO
# used for testing the code that reads from input
from unittest.mock import patch, mock_open
import numpy as np
import time

class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


timeout = 10  # seconds

def run_test(prob_path:str=None, problem_list:List[str]=None, prob_index:int=None, 
        test:str=None, debug:bool=False, example_tests:bool=False):
    """
    if test is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    
    
    #确认问题路径
    if prob_path is None and problem_list is None:
        print("please provide either prob_path or problem_list")
        exit()

    if debug:
        print(f"start = {datetime.now().time()}")
    if prob_path is not None:
        root = prob_path #'/home2/nsy/APPS/APPS/train/1535'
    elif problem_list is not None:
        root = problem_list[prob_index]

    if os.path.exists(os.path.join(root, "input_output.json")):
        with open(os.path.join(root, "input_output.json")) as f:
            in_outs = json.load(f)
            if debug:
                print(f"test cases json = {in_outs['inputs']} {in_outs['outputs']}")
            
            if in_outs.get("fn_name") is None:
                which_type = CODE_TYPE.standard_input  # Standard input
                method_name = None
            else:
                which_type = CODE_TYPE.call_based  # Call-based
                method_name = in_outs["fn_name"]
    elif not example_tests:
        return [], [], [], None 
    elif example_tests: 
        which_type = CODE_TYPE.standard_input  # assuming this method type 
        method_name = None
    
    if example_tests:
        if os.path.exists(os.path.join(root, "example_input_output.json")):
            with open(os.path.join(root, "example_input_output.json")) as f:
                in_outs = json.load(f)
                if in_outs is None: 
                    return [], [], [], None 
        else:
            return [], [], [], None
    
    if debug:
        print(f"loaded json = {datetime.now().time()}")
    
    #else:
    #    continue
    if test is None:
        return [], [], [], None 
    #如果代码不为空
    elif test is not None:
        results = []
        errors = []
        outputs = []
        runtimes = []
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        if debug:
            print(f"loading test code = {datetime.now().time()}")
        
        #代码中有方法名
        if which_type == CODE_TYPE.call_based:
            sol += test
            if debug: # or True:
                print(f"sol = {sol}")
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                if "class Solution" not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            
            except Exception as e:
                signal.alarm(0)
                if debug: 
                    print(f"type 0 compilation error = {e}")
                results.append(-2)
                errors.append(e)
                outputs.append(None)
                return results, errors, outputs, sol
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            # sol
            tmp_test = test.split("\n")

            new_test = []
            for x in tmp_test:   
                if (not x.startswith("from ")) and (not x.startswith("import ")):  
                    new_test.append("\t" + x + "\n")   
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test   
          
            new_test = ""
            started = False
            for i in tmp_test:
                if i.startswith("\t") and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))): 
                    new_test += "\t" + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f"sol = {sol}")
                # print(f"{o}") 
            method_name = "code"
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug: 
                    print(f"type 1 compilation error = {e}")
                results.append(-2)
                errors.append(e)
                outputs.append(None)
                return results, errors, outputs, sol
            signal.alarm(0)
        if debug:
            print(f"get method = {datetime.now().time()}")

        try:
            method = getattr(tmp, method_name)  # get_attr second arg must be str
        except Exception as e:
            signal.alarm(0)
            # e = sys.exc_info()
            print(f"unable to get function error = {e}")
            results.append(-2)
            errors.append(e)
            return results,errors
        
        #for index, inputs in enumerate(in_outs["inputs"]):
        for index, inputs in tqdm(enumerate(in_outs["inputs"]), total=len(in_outs["inputs"]), ncols=0, leave=False): 
            
            gc.collect()
            
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k,v in inputs[0].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [{int(k): v for k,v in in_outs["outputs"][index].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [{int(k): v for k,v in in_outs["outputs"][index][0].items()}]
            except:
                True

            if debug:
                print(f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}")
            if which_type == CODE_TYPE.call_based:  # Call-based
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    # print("------------")
                    # print(inputs)
                    begin_time = time.time()
                    output = method(*inputs)
                    runtime = round(1000000*(time.time() - begin_time), 6)
                    #runtimes.append(runtime)
                    
                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)
                    
                    tmp_result = output == in_outs["outputs"][index]
                    if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                        tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
                    except:
                        True
                    results.append(tmp_result)
                    errors.append(None)
                    outputs.append(output)
                    if tmp_result == True:
                        runtimes.append(runtime)

                    # reset the alarm
                    signal.alarm(0)
                    
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    if debug: 
                        print(f"Standard input runtime error or time limit exceeded error = {e}")
                    results.append(-1)
                    errors.append(e)
                    outputs.append(None)
                    
                    ## TESTING TRICK: exit loop if not pass a test case 
                    return results, errors, outputs, sol
                    #continue
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
            
            elif which_type == CODE_TYPE.standard_input:  
                faulthandler.enable()
                signal.alarm(timeout)  
                passed = False

                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)   
                if isinstance(in_outs['outputs'][index], list):
                    in_outs['outputs'][index] = "\n".join(in_outs['outputs'][index])

                with Capturing() as output:
                    try:
                        signal.alarm(timeout) 
                        runtime = call_method(method, inputs)
                        #runtimes.append(runtime)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        if debug:
                            print(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                        results.append(-1)
                        errors.append(e) 
                        outputs.append(None)
                        ## TESTING TRICK: exit loop if not pass a test case 
                        return results, errors, outputs, sol
                    
                    signal.alarm(0)
                
                if not passed:
                    if debug:
                        nl = "\n"
                        if not isinstance(inputs, list):
                            print(f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                        else:
                            print(f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    continue

                if passed and debug:
                    print(f"==> output = {output}, test outputs = {in_outs['outputs'][index]}")

                if custom_compare_(output, in_outs['outputs'][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    errors.append(None)
                    outputs.append(output)
                    runtimes.append(runtime)
                    continue  

                # ground truth sequences are expressed as lists not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = False
                try:
                    tmp_result = (output == [in_outs["outputs"][index]])
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        if isinstance(output[0], str):
                            tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
                except Exception as e:
                    if debug: 
                        print(f"Failed check1 exception = {e}")
                    pass

                if tmp_result == True:  
                    results.append(tmp_result)
                    errors.append(None)
                    outputs.append(output)
                    runtimes.append(runtime)
                    continue

                # try one more time without \n
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        in_outs["outputs"][index][tmp_index] = [x.strip() for x in in_outs["outputs"][index][tmp_index] if x]
                else:
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                    in_outs["outputs"][index] = list(map(lambda x:x.strip(), in_outs["outputs"][index]))

                try:
                    tmp_result = (output == [in_outs["outputs"][index]])
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug: 
                        print(f"Failed check2 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    errors.append(None)
                    outputs.append(output)
                    runtimes.append(runtime)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    output = list(filter(len, output))

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}") 
                    else:
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}") 
                
                if tmp_result == True:
                    results.append(tmp_result)
                    errors.append(None)
                    outputs.append(output)
                    runtimes.append(runtime)
                    continue

                try:
                    tmp_result = (output == [in_outs["outputs"][index]])
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug: 
                        print(f"Failed check3 exception = {e}")
                    pass
                
                output_float = output
                try:
                    output_float = [float(e) for e in output]
                    gt_float = [float(e) for e in in_outs['outputs'][index]]
                    tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                except Exception as e:
                    pass
                try:
                    if isinstance(output[0], list):
                        output_float = [float(e) for e in output[0]]
                        gt_float = [float(e) for e in in_outs['outputs'][index][0]]
                        tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                except Exception as e:
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    errors.append(None)
                    outputs.append(output_float)
                    runtimes.append(runtime)
                    continue

                # try by converting the stuff into split up list
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                else:
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                try:
                    tmp_result = (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug: 
                        print(f"Failed check4 exception = {e}")
                    continue

                if tmp_result == True:
                    results.append(tmp_result)
                    errors.append(None)
                    outputs.append(output)
                    runtimes.append(runtime)
                    continue 

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = set(i)    
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)

                try:
                    tmp_result = (set(frozenset(s) for s in output) == set(frozenset(s) for s in in_outs["outputs"][index]))
                except Exception as e:
                    if debug: 
                        print(f"Failed check5 exception = {e}")

                # if they are all numbers, round so that similar numbers are treated as identical
                try:
                    tmp_result = tmp_result or (set(frozenset(round(float(t),3) for t in s) for s in output) ==\
                        set(frozenset(round(float(t),3) for t in s) for s in in_outs["outputs"][index]))
                except Exception as e:
                    if debug: print(f"Failed check6 exception = {e}")
                
                if tmp_result == True and debug:
                    print("PASSED")
 
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                if tmp_result == True:
                    runtimes.append(runtime)
            
                if tmp_result != True:
                    ## TESTING TRICK: exit loop if not pass a test case 
                    return results, errors, outputs, sol
                
                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    else:
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}") 
                        
    return results, errors, outputs, sol, runtimes

def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # TODO: the below line was originally commented 
    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split("\n"))
    @patch('sys.stdin.read', lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            begin_time = time.time()
            _method()
            runtime = round(1000000*(time.time() - begin_time), 6)
        except SystemExit as e:
            runtime = 'runtime error'
            pass
        finally:
            pass
        return runtime
    return _inner_call_method(method)

def custom_compare_(output, ground_truth):
    
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False

def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2 