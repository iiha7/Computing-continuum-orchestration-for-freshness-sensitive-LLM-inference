# Computing Continuum Orchestration for Freshness-Sensitive LLM Inference

This repository contains the implementation of the **freshness-aware orchestration framework** presented in our paper on LLM-assisted inference over the IoT–edge–cloud computing continuum.

The framework dynamically selects between **edge and cloud LLM inference** for transmitted IoT requests. It jointly considers response quality, Age of Processing (AoP), freshness deadlines, communication load, task importance, and edge–cloud path conditions. A lightweight IoT pre-transmission gate further suppresses low-value updates before they are transmitted to the edge or cloud.

---

## Framework Overview

The system consists of three main layers:

```text
IoT Layer
    │
    │  Lightweight pre-transmission gate
    │
    ▼
Edge Layer
    │
    │  Freshness-aware orchestration
    │  ┌───────────────────────┐
    ├──► Compact Edge LLM      │
    │  └───────────────────────┘
    │
    │  Selective cloud offloading
    ▼
Cloud Layer
       Large Cloud LLM
```

### IoT layer

IoT devices generate sensing updates and apply a lightweight pre-transmission gate based on a risk/criticality metric. Low-risk updates can be handled locally without transmission, reducing unnecessary traffic.

### Edge layer

Transmitted requests reach the edge orchestration layer, which maintains the task, model, network, and cloud-budget context. The orchestration policy selects either:

* **Edge:** execute using the compact edge LLM.
* **Cloud:** forward the request to the larger cloud LLM.

### Cloud layer

The cloud provides a higher-capability LLM but introduces additional edge–cloud communication delay and cloud usage. Cloud execution is therefore treated as a resource that should be used selectively rather than continuously.

---

## Key Components

The final framework combines:

* **IoT pre-transmission gating**
* **Compact edge LLM inference**
* **Large cloud LLM inference**
* **Offline LLM profiling**
* **Age of Processing (AoP)-aware orchestration**
* **Freshness-weighted response quality (FWQ)**
* **Adaptive cloud budgeting**
* **Edge–cloud path-condition awareness**
* **Dueling Double DQN-based policy learning**

The learned policy determines the execution location for each transmitted request while respecting the adaptive cloud-use budget.

---

## Decision Context

The orchestration state includes information from three categories:

### Task context

* Prompt complexity
* Task criticality/value
* Freshness deadline
* Request characteristics

### Model/profile context

* Profiled edge response quality
* Profiled cloud response quality
* Profiled inference latency
* Estimated response size

### Network and budget context

* IoT access-link conditions
* Edge–cloud path condition
* Cloud latency
* Remaining cloud budget
* Previous execution action

---

## LLM Profiling

The framework does not rely on self-reported LLM confidence.

Instead, the edge and cloud models are profiled offline to characterize their behavior for the considered task types. The resulting profiles provide estimates of:

* Response quality
* Processing latency
* Response size

These profiles are subsequently used by the orchestration framework when making edge/cloud execution decisions.

> **Note:** The exact edge and cloud LLM configurations, model names, prompts, and profiling parameters are documented in the corresponding implementation files and experiment configuration.

---

## Optimization Objective

For each transmitted request, the framework balances response quality and freshness against communication and execution overhead.

The reward considers:

* Structured response quality
* Task importance
* Freshness utility
* Age of Processing
* Deadline violations
* Edge–cloud communication load
* Unnecessary switching between execution locations

Cloud usage is additionally constrained by an adaptive budget whose allowable fraction changes according to the observed edge–cloud path condition.

---

## Execution Policy

The action space is:

```text
edge
cloud
```

The framework uses **Dueling Double DQN** to learn the orchestration policy.

The policy is trained using simulated IoT requests and network conditions and subsequently evaluated under different edge–cloud delay conditions.

---

## Experimental Evaluation

The evaluation compares three configurations:

| Configuration | Description                                                              |
| ------------- | ------------------------------------------------------------------------ |
| **EdgeOnly**  | All transmitted requests are processed using the compact edge LLM.       |
| **CloudOnly** | All transmitted requests are forwarded to the cloud LLM.                 |
| **Ours**      | The learned policy dynamically selects between edge and cloud execution. |

All configurations use the same IoT pre-transmission gating mechanism.

The evaluation considers:

* Response quality ($Q$)
* Freshness-weighted quality (FWQ)
* Age of Processing (AoP)
* Cloud usage
* Edge–cloud communication load
* Deadline satisfaction
* Execution-path distribution

Network conditions are evaluated under multiple edge–cloud stress levels, including nominal, delay-stress, and severe-delay conditions.

---

## Reproducing the Experiments

### 1. Clone the repository

```bash
git clone <[REPOSITORY-URL](https://github.com/iiha7/Computing-continuum-orchestration-for-freshness-sensitive-LLM-inference)>
cd <Computing-continuum-orchestration-for-freshness-sensitive-LLM-inference>
```

### 2. Create the environment

```bash
python -m venv .venv
```

Activate it:

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the experiment

Experiment parameters, model configurations, network conditions, and training settings are specified in the configuration files.


### 5. Run the evaluation pipeline

```bash
bash run_full_pipeline.sh
```

The evaluation produces per-request and aggregated results for the EdgeOnly, CloudOnly, and other proposed configurations.

---

## Results Analysis

The analysis produces:

```text
paper_results/
├── summary_metrics_mean_over_seeds.csv
├── summary_metrics_by_seed.csv
├── table_relative_tradeoff.csv
├── table_nominal_results.csv
├── table_aop_transmitted.csv
├── action_mix_ours.csv
├── action_heatmap_ours.csv
├── fig_ieee_drl_action_mix.pdf
├── fig_ieee_drl_action_mix.png
├── fig_action_heatmap.pdf
└── fig_action_heatmap.png
```

The generated tables can be directly used to update the numerical results reported in the manuscript.

---

## Repository Structure

The final repository is organized approximately as follows:

```text
.
├── README.md
├── requirements.txt
│
├── configs/
│   ├── ...
│
├── data/
│   ├── ...
│
├── src/
│   ├── ...
│
└── tools/
    ├── ...
```

---

## Reproducibility

All reported experiments should be run using the same:

* Dataset/request configuration
* Model configurations
* Network traces
* Freshness deadlines
* Training hyperparameters
* Evaluation settings
* Random seeds
---

## Citation

If you use this framework or code in your research, please cite the associated paper:

```bibtex
@inproceedings{<citation-key>,
  title     = {Computing Continuum Orchestration for Freshness-Sensitive LLM Inference},
  author    = {Abu Ali, Hamzeh and Kizilkaya, Burak and Pezaros, Dimitrios},
  booktitle = {IEEE International Conference on Cloud Networking (CLOUDNET)},
  year      = {2026}
}
```

The citation information will be updated with the final publication details.

---

## License

This repository is released for research and academic use. The license and additional usage conditions will be specified upon release.
