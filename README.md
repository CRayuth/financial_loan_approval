# Financial Approval Agent

An AI agent that reviews credit card applications and decides to approve, deny, or escalate for manual review.

Dataset: [Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Train the model** (run once to generate model files)

```bash
python -m think.train
```

**2. Evaluate the model** (optional — prints metrics and saves confusion matrix)

```bash
python -m think.evaluate
```

**3. Run the app**

```bash
streamlit run act/app.py
```

