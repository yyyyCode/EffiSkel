import json
import numpy as np
import os
import os.path 
import pprint
import glob 
from tqdm import tqdm
import pdb
import traceback 
import pickle as pkl 
# import dill as pkl
from typing import List
import multiprocessing
from testing_util import run_test
import statistics

TIMEOUT = 4
ITERATIONS = 30

def check_correctness(prob_path, generation, timeout, debug , example_tests):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(prob_path, generation, debug,example_tests, result_result,result_error,result_code,result_runtime):
        tmp = run_test(prob_path=prob_path, test=generation, debug=debug, example_tests = example_tests )
        result_result.extend(tmp[0])
        result_error.extend(tmp[1])
        result_code.extend(tmp[3])
        if (len(tmp) < 5):
            result_runtime.append(0)
        else:
            result_runtime.extend(tmp[4])
    manager = multiprocessing.Manager()
    result_result = manager.list()
    result_error = manager.list()
    result_code = manager.list()
    result_runtime = manager.list()
    
    p = multiprocessing.Process(target=_temp_run, args=(prob_path, generation, debug, example_tests,result_result,result_error,result_code,result_runtime))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result_result:
        # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead 
        result_result = [-1]
        result_error = [None]
        if debug:
            print(f"global timeout")
        return result_result,result_error
    return result_result,result_error,result_code,result_runtime



def eval_and_save_problems(args):

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    problem_id = args.index
    real_index = str(problem_id).zfill(4)
    print('Testing sample {}'.format(real_index))
    
    if args.example_tests:
        print("Using example tests") 
    
    problem = args.test_path + '/{}/'.format(real_index) 
    codes_loc = args.code_path + '/{}'.format(str(problem_id)) + '.json'

    if not os.path.isfile(codes_loc):      
        exit() 
    with open(codes_loc, "r") as file: 
        gen_codes = json.load(file)
    
    gen_codes_id = gen_codes[str(problem_id)]
    gen_codes_list = gen_codes_id['codes']

    test_file = os.path.join(problem, "input_output.json")   
    if not os.path.isfile(test_file):      
        exit()
    tests = json.load(open(test_file, 'r'))
    nb_tests = len(tests['inputs'])
    if args.max_tests!=-1 and nb_tests > args.max_tests: 
        exit() 

    all_results, all_errors,all_outputs, all_sols = [], [], [], []
    all_runtimes = []

    for o_idx, o in tqdm(enumerate(gen_codes_list), total=len(gen_codes_list), ncols=0, leave=False):   
        if args.debug:
            print("\ncandidate idx: ",o_idx,"========================================================================================\n")
        curr_results = []
        curr_errors = []
        curr_outputs = []
        curr_codes = []
        curr_runtimes = []
        avg_curr_runtimes = []
        for _ in range(ITERATIONS):
            try:
                curr_results, curr_errors,curr_codes, curr_runtimes = check_correctness(prob_path=problem,generation=o,timeout=TIMEOUT, debug=args.debug, 
                                            example_tests=args.example_tests)
                fixed = []
                for e in curr_results:
                    if isinstance(e, np.ndarray):
                        if len(e)==0:
                            e = -2
                            fixed.append(e)
                            continue
                        e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_results = fixed

            except Exception as e:
                print(f"test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_results, list)
                avg_curr_runtimes.append(calculate_average(curr_runtimes))
                #print(curr_runtimes, avg_curr_runtimes)
                # all_errors.append(curr_errors)
        all_results.append(curr_results)
        all_errors.append(curr_errors)
        all_outputs.append(curr_outputs)
        all_sols.append(curr_codes)
        all_runtimes.append(calculate_average(avg_curr_runtimes))
    save_times = {'times':all_runtimes}
    output = args.output_path + '/{}/'.format(real_index) + 'times.json'
    if not os.path.exists(args.output_path + '/{}/'.format(real_index)):
        os.makedirs(args.output_path + '/{}/'.format(real_index))
    with open(output, 'w') as f:
        json.dump(save_times, f)
    print(save_times)
    '''
    How to read results:
    [-2] = compile error, 
    [-1] = runtime error 
    [False] = failed test case 
    [True] = passed test case
    '''

def calculate_average(float_list):
    if float_list:  
        return round(sum(float_list) / len(float_list), 6)
    else:
        return 0              

def main(args):    
    argsdict = vars(args)    
    eval_and_save_problems(args)

if __name__ == "__main__":
    from unit_test_configs import * 
    main(args)
