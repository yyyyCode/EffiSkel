# EffiSkel

EffiSkel is a high-efficiency code generation framework with structured skeleton supervision.
This repository contains code, data, and models related to the ASE 2025 paper: "Chiseling Out Efficiency: Structured Skeleton Supervision for Efficient Code Generation".

---

## 📁 Project Structure

<pre>
EffiSkel/
├── configs/ # Training and Inference Setup
├── data/ # Benchmark datasets
├── Datasets/ # Datasets processing
├── evaluate/ # Used to evaluate the functional correctness and efficiency of the code
├── trainer/ 
├── transformers/ 
├── generate.py/
├── train.py/
├── requirement.py
└── README.md
</pre>
  
---

## 🚀 Usage

### 1. Installation

Please follow the requirements.txt file to install the relevant dependencies or run:

<pre> pip install -r requirements.txt</pre>

Since our method modifies the transformers of huggingface, please make sure to install the same transformers as ours (we use transformers version 4.44.2):

<pre>
cd transformers
pip install -e .
</pre>
  
### 2. Datasets

We used two datasets

APPS
EffiBench

### 3. Finetuning




