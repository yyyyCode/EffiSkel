import torch
import random
import gc
import os
import io
import transformers
from Datasets.reindent import run as run_reindent
from tqdm import tqdm 
import json

class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, mode, max_tokens, sample_mode):
        self.dataroot = dataroot 
        self.problem_dirs = problem_dirs 

        self.mode = mode
        self.sample_mode = sample_mode # Either "uniform_sol" or "uniform_prob"
        self.max_tokens = max_tokens

        self.samples = []           # Should be set in initialize()
        self.initialize()
        print("===================================================================================")
        print("load tokenizer:",mode)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("")

    def load_gt_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        
        for sol_str in sols:
            sol_str = reindent_code(sol_str)
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample)
        
        return samples

    def initialize(self):
        """
        Assume self.dataroot is set to folderName/data
        """

        all_samples = []
        skipped_problems = []

        all_samples_dict = {} # Mapping from question_fname to list of samples

        efficientCodeIndex_dict_json = ""
        with open(efficientCodeIndex_dict_json,'r',encoding='utf-8') as eci:
            efficientCodeIndex_dict = json.load(eci)

        print(f"Loading {len(efficientCodeIndex_dict)} problems from {self.dataroot}.")
        for key, value in tqdm(efficientCodeIndex_dict.items()):
            sols_str_list = []
            question_fname = os.path.join(self.dataroot, key, "question.txt")
            sols_fname = os.path.join(self.dataroot, key, "solutions.json")
            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(key)
                continue
            # Read the question description
            with open(question_fname, 'r') as q:
                question_str = q.read()
            
            with open(sols_fname, 'r',encoding='utf-8') as s:
                sols_str = json.load(s)

            starter_code = os.path.join(self.dataroot, key, "starter_code.py")

            # print(question_fname)

            if os.path.exists(starter_code):
                answer_type = "\nUse Call-Based format\n"
            else:
                answer_type = "\nUse Standard Input format\n"

            if (os.path.isfile(starter_code)):
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                starter_code = ""

            for i in value:
                code = sols_str[i]
                sols_str_list.append(code)
            gt_samples = self.load_gt_samples(sols_str_list, answer_type, starter_code, question_str)
            all_samples += gt_samples
        
        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
        print(all_samples[0])
        self.samples = all_samples
        self.samples_dict = all_samples_dict


    def __len__(self):
        return len(self.samples)


    def pack_samples(self, idx):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the  
        self.question_prefix. These will be added later and the total input will be 
        truncated if necessary.

        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = [] 

        if self.sample_mode == 'uniform_sol':
            curr_q, curr_s, curr_a, curr_q_prefix = self.samples[idx]
        elif self.sample_mode == 'uniform_prob':
            curr_q = random.choice(list(self.samples_dict.keys()))
            curr_q, curr_s, curr_a, curr_q_prefix = random.choice(self.samples_dict[curr_q])
        else:
            raise NotImplementedError()

        while curr_num_tokens < self.max_tokens:

            # Never remove. Fixes stalling bug.
            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))

            curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))

            if self.sample_mode == 'uniform_sol':
                curr_q, curr_s, curr_a, curr_q_prefix = random.choice(self.samples)
            elif self.sample_mode == 'uniform_prob':
                curr_q = random.choice(list(self.samples_dict.keys()))
                curr_q, curr_s, curr_a, curr_q_prefix = random.choice(self.samples_dict[curr_q])
            else:
                raise NotImplementedError()

        return curr_samples

    def __getitem__(self, idx):
        
        raw_samples = self.pack_samples(idx)

        retval = sample_gpt_task(
                raw_samples,
                max_tokens=self.max_tokens, 
                tokenizer=self.tokenizer, 
            )
    
        gc.collect()
        return retval

def sample_gpt_task(raw_samples, max_tokens, tokenizer):
    """
    Create the true sample used for the GPT task
    """

    input_ids = []
    label_ids = []
    
    for q_str, s_str, a_str, answer_type in raw_samples:
        
        # Loss is not calculated on this
        q_str =  "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"

        question_token_ids = tokenizer.encode(q_str, verbose=False)
        answer_token_ids = tokenizer.encode(a_str, verbose=False, add_special_tokens=False)
        answer_token_ids.append(tokenizer.eos_token_id)

        input_ids.extend(question_token_ids)
        input_ids.extend(answer_token_ids)
        
        label_ids.extend([-100] * len(question_token_ids))
        label_ids.extend(answer_token_ids)
    
    # Sanity check
    assert len(input_ids) == len(label_ids)

    if len(input_ids) < max_tokens:
        print(len(input_ids))
        import pdb; pdb.set_trace()

    # Cut off the excess
    input_ids = input_ids[:max_tokens]
    label_ids = label_ids[:max_tokens]

    return {
        "input_ids" : torch.LongTensor(input_ids),
        "labels" :  torch.LongTensor(label_ids)
    }


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
