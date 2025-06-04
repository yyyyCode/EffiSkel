# EffiSkel

⚡️ EffiSkel is a high-efficiency code generation framework with structured skeleton supervision.
This repository contains code, data, and models related to the ASE 2025 paper: "Chiseling Out Efficiency: Structured Skeleton Supervision for Efficient Code Generation".

![EffiSkel Framework](assets/images/EffiSkel.png)

<details>
  <summary>Contributions</summary>

  - 1 Conceptual Innovation. We propose the concept of an efficiency skeleton to highlight structural aspects that strongly influence code efficiency. While efficiency also depends on external factors (e.g., hardware or compilers), we focus on structural properties as they offer actionable, learnable signals for LLMs. By using these patterns as explicit supervision—rather than relying solely on code examples—we guide models to encode algorithmic best practices and performance-aware programming more effectively.
  - 2 Technical Advances. We propose three complementary strategies to systematically extract representative efficiency skeletons. Moreover, we develop a structure-aware multi-task learning framework that jointly optimizes skeleton prediction and code generation, effectively embedding efficiency insights directly into LLM training.
  - 3 Empirical Validation. We introduce the APPS+EFFI benchmark, explicitly focusing on efficiency-critical code generation tasks, and demonstrate through extensive experiments that EffiSkel achieves significant improvements in runtime efficiency across multiple benchmarks.

</details>

---

## 📁 Project Structure

<pre>
EffiSkel/
├── configs/ # ⚙️ Training and Inference Setup
├── data/ # 📊 Benchmark datasets
├── Datasets/ # 📦 Datasets processing
├── evaluate/ # 📝 Evaluate code correctness & efficiency
├── trainer/ # 🎯 Training launcher
├── transformers/ # 🧩 Model backbone and customization
├── generate.py/ # 🚀 Generation code
├── train.py/ 🏋️ Model training
├── requirement.py/ # 📋 Project requirements
└── README.md/ # 📖 Project documentation
</pre>
  
---

## 🧰 Installation 

Please follow the requirements.txt file to install the relevant dependencies or run:

<pre> pip install -r requirements.txt</pre>

Since our method modifies the transformers of huggingface, please make sure to install the same transformers as ours (we use transformers version 4.44.2):

<pre>
cd transformers
pip install -e .
</pre>
  
## 📚 Datasets

We used three datasets: APPS & EffiBench & APPS+EFFI.

You can download the APPS dataset [here](https://github.com/hendrycks/apps) and EffiBench [here](https://github.com/huangd1999/EffiBench).
You can download the APPS+EFFI dataset from the [data](data/APPS+EFFI) folder.

## 🤗 Model
We fine-tune on four models

  - [Qwen2.5-Coder (1.5B)](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
  - [StarCoder2 (3B)](https://huggingface.co/bigcode/starcoder2-3b)
  - [DeepSeek-Coder (6.7B)](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)
  - [CodeLlama (7B)](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)

## 🏋️ Finetuning

First, fine-tune the base model on the code of the APPS+EFFI dataset and the corresponding natural language description of the APPS dataset by running the following code:
<pre>
python train_base_model.py
</pre>
Then, fine-tune the base model in a multi-task framework :
<pre>
python train_mask_model.py
python train_skeleton_model.py
python train_total_model.py
</pre>

## ✨ Generating

Generate candidate codes for different fine-tuning methods:
<pre>
python generate_base.py
python generate_mask.py
python generate_skeleton.py
python generate_total.py
</pre>

## 📊 Evaluate

You can run "test_one_solution.sh" to evaluate the functional correctness and efficiency of the generated code:
<pre>
cd evaluate/metric
bash test_one_solution.sh
cd evaluate/metric_time
bash test_one_solution.sh
</pre>




