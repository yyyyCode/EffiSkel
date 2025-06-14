import argparse

parser = argparse.ArgumentParser(description="Run model to generate Python programs.")

parser.add_argument("-t","--test_path", default="/data/APPS/test", type=str, help='Path to test samples')
parser.add_argument("--output_path", default=None, type=str, help='Path to save output programs')
parser.add_argument("--model_path", default=None, type=str, help='Path of trained model')
parser.add_argument("--tokenizer_path", type=str, help='Path to the tokenizer')  

parser.add_argument("--num_seqs", default=10, type=int, help='Number of total generated programs per test sample')
parser.add_argument('--num_seqs_per_iter', default=10, type=int, help='Number of possible minibatch to generate programs per iteration, depending on GPU memory')

parser.add_argument("--max_len", default=1600, type=int, help='Maximum length of output sequence') 
parser.add_argument("--temperature", default=0.8, type=float, help='temperature for sampling tokens')
parser.add_argument("--codeskeleton_head", default=True, type=bool, help="")
parser.add_argument("--is_codeskeleton", default=False, type=bool, help="")
parser.add_argument("-s","--start", default=0, type=int, help='start index of test samples')
parser.add_argument("-e","--end", default=5000, type=int, help='end index of test samples')

args = parser.parse_args()

