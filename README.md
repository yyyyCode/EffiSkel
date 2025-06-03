# EffiSkel

⚡️ EffiSkel is a high-efficiency code generation framework with structured skeleton supervision.
This repository contains code, data, and models related to the ASE 2025 paper: "Chiseling Out Efficiency: Structured Skeleton Supervision for Efficient Code Generation".

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

We used two datasets:

    APPS

    EffiBench

You can download the APPS dataset here and EffiBench here.

## 🏋️ Finetuning

(Instructions for fine-tuning models.)

## ✨ Generating

(Instructions for generating code using the trained models.)

## 📊 Evaluate

You can run 'test_one_solution.sh' to evaluate generated codes:

<pre>
bash test_one_solution.sh
python eval_metric.py
</pre>




