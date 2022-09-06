# LogPPT

Repository for the paper: Log Parsing with Prompt-based Few-shot Learning

**Abstract:** Logs generated by large-scale software systems provide crucial information for engineers to understand the
system status and diagnose problems of the systems. Log parsing, which converts raw log messages into structured data,
is the first step to enabling automated log analytics. Existing log parsers extract the common part as log templates
using statistical features. However, these log parsers often fail to identify the correct templates and parameters
because: 1) they often overlook the semantic meaning of log messages, and 2) they require domain-specific knowledge for
different log datasets. To address the limitations of existing methods, in this paper, we propose LogPPT to capture the
patterns of templates using prompt-based few-shot learning. LogPPT utilises a novel prompt tuning method to recognise
keywords and parameters based on a few labelled log data. In addition, an adaptive random sampling algorithm is designed
to select a small yet diverse training set. We have conducted extensive experiments on 16 public log datasets. The
experimental results show that LogPPT is effective and efficient for log parsing.

## I. Framework

<p align="center"><img src="docs/images/LogPPT_overview.png" width="500"><br>An overview of LogPPT</p>

LogPPT consists of the following components:
1. **Adaptive Random Sampling algorithm**:  A few-shot data sampling algorithm, which is used to select K labelled logs for training (K is small).
2. **Virtual Label Token Generation**: A module to generate a virtual label token (I-PAR) for prompt tuning.
3. **Prompt-based Parsing**: A module to tune a pre-trained language model using prompt tuning for log parsing

## II. Requirements
### 2.1. Library
1. Python 3.8
2. torch
3. transformers
4. ...

To install all library:
```shell
$ pip install -r requirements.txt
```

### 2.2. Pre-trained models
To download the pre-trained language model:
```shell
$ cd pretrained_models/roberta-base
$ bash download.sh
```

## III. Usage:

### 3.1. Few-shot data Sampling

### 3.2. Training & Parsing

## Results
### RQ1: Parsing Effectiveness
- Accuracy:
<p align="center"><img src="docs/images/RQ1_comparison.png"></p>
- Robustness:
<p align="center"><img src="docs/images/RQ1_robustness1.png" width="700"><br>Robustness across different log data types</p>
<p align="center"><img src="docs/images/RQ1_robustness2.png" width="500"><br>Robustness across different numbers of training data</p>
- Accuracy on Unseen Logs:
<p align="center"><img src="docs/images/RQ1_unseen.png" width="500"><br>Accuracy on Unseen Logs</p>

### RQ2: Runtime Performance Evaluation
<p align="center"><img src="docs/images/RQ2_runtime.png" width="500"><br>Running time of different log parsers under different volume</p>

### RQ3: Ablation Study
- We exclude the Virtual Label Token Generation module and let the pre-trained model automatically assign the embedding for the virtual label token “I-PAR”. To measure the contribution of the Adaptive Random Sampling module, we remove it from our model and randomly sample the log messages for labelling.
<p align="center"><img src="docs/images/RQ3_ablation_study.png" width="500"><br>Ablation Study Results</p>

- We vary the number of label words from 1 to 16 used in the Virtual Label Token Generation module.
<p align="center"><img src="docs/images/RQ3_lbl_words.png" width="500"><br>Results with different numbers of label words</p>

### RQ4: Comparison with Different Tuning Techniques
We compare LogPPT with fine-tuning, hard-prompt, and soft-prompt.
- Effectiveness:
<p align="center"><img src="docs/images/RQ4_accuracy.png" width="500"><br>Accuracy across different tuning methods</p>
- Efficiency:
<p align="center"><img src="docs/images/RQ4_parsingtime.png" width="500"><br>Parsing time across different tuning methods</p>