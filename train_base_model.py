import os
import pprint
import json
import transformers
import torch
from Datasets.APPSBaseDataset import APPSBaseDataset
from transformers import Trainer  
from peft import get_peft_model, LoraConfig, TaskType
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

def run_training(args, train_data):

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16,
        pad_token_id=tokenizer.eos_token_id)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print('Finished loading model {}'.format(args.model))

    start_iteration = 0
    train_data.start_iteration = start_iteration
    print(f"Starting main loop")

    training_args = transformers.TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,

        do_train=True,
        do_eval=False,
        do_predict=True,
        evaluation_strategy='no',
        eval_steps=0,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        lr_scheduler_type='constant_with_warmup',

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=args.save_freq,
        save_total_limit=args.save_total_limit,

        dataloader_drop_last=True,
        dataloader_num_workers=0 if args.db else 8,

        local_rank=args.local_rank,

        deepspeed=args.deepspeed,
        fp16=args.fp16,

    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.train()


    if args.local_rank == 0:
        model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))
        
def get_dataset(args):

    fnames = os.listdir(args.train_path)

    # train in debugging mode with small data split
    if args.db:
        fnames = fnames[:50]

    train_data = APPSBaseDataset(
        dataroot=args.train_path,
        problem_dirs=fnames,
        mode=args.model,
        max_tokens=1600,
        sample_mode=args.sample_mode,
    )

    return train_data

def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    train_data = get_dataset(args)
    
    # Save args to file
    json.dump(argsdict, open(os.path.join(args.save_dir, "args.json"), 'w'))

    # Load and train model; save model checkpoints
    run_training(args, train_data)


if __name__ == "__main__":
    from configs.train_base_configs import *
   
    main(args)