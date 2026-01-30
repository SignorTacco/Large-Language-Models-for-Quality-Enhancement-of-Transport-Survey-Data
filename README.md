# Large-Language-Models-for-Quality-Enhancement-of-Transport-Survey-Data

This repository contains the research and implementation code for a thesis exploring two distinct methodologies to resolve errors in self-reported travel data within the **Danish National Travel Survey (TU)**. 

The project investigates how traditional string-matching and modern Generative AI (RAG) compare when resolving the "Semantic Gap" between human narrative and machine-readable coordinates.

## 🔬 Experimental Framework: Two Approaches
Instead of a single pipeline, this study developed and compared two independent methods to address data loss:

### Method A: The Deterministic Approach (DET)
* **Focus:** Morphological errors (typos, phonetic misspellings, abbreviations).
* **Mechanism:** Strict mathematical string matching using Levenshtein edit distance against the Danish Address Web Architecture (DAWA) registry.
* **Result:** Highly efficient at resolving ~83% of standard noise but failed completely at the "Lexical Cliff" (inputs with zero character overlap).

### Method B: The Semantic Approach (RAG)
* **Focus:** Ontological errors (logical artifacts, conceptual descriptions, relative locations).
* **Mechanism:** Retrieval-Augmented Generation (RAG) using local vector embeddings and a quantized Llama-3-8B model.
* **Result:** Acted as a "Semantic Bridge," successfully recovering ~1,500 trips that the deterministic method could not see (e.g., "The gym by the station").



## 📈 Analysis & Psychophysics
The repository includes the analysis of the "Unified Bias Matrix," which uses the combined valid results of both experiments to quantify human reporting error:
* **Distance Perception:** Short trips (<2km) are overestimated by 64.3% (**Short Trip Inflation**).
* **Time Perception:** Long trips (>60m) are underestimated by 89.1% (**Optimism Bias**).

## 🛡️ Privacy & Local Execution
Both experiments were designed to meet the strict **GDPR requirements** of the Danish National Transport Model.
* No data leaves the local environment.
* All LLM inference is performed on-premise using quantized open-source weights.
* Vector search is handled in-memory via FAISS.





## 🛠️ Implementation Workflow & File Structure
The project follows a "Golden Cascade" logic: preprocessing the raw survey data, executing a high-volume Deterministic experiment locally, and running a high-reasoning Semantic (RAG) experiment on the DTU HPC Cluster.

### 1. Infrastructure & Reference Data
Before running the pipelines, the environment and the Danish address baseline must be established.

* Addresses/fetch_all_denmark_csv.py: Downloads the official Danish address registry (DAWA).

* Addresses/build_index.py: Processes raw addresses into searchable artifacts (stored in addr_index/).

* Setting_up_docker_osm.txt: Technical instructions for hosting the local OSRM (Open Source Routing Machine) instance via Docker.

### 2. Preprocessing: The "Efterkod" Alignment
This phase aligns the raw survey data with the official ground truth (Efterkod) to identify the errors that need fixing.

* Individual Playbooks: Process raw session, trip, and stage data into TuSession_Edited_Nicola.csv, EfterkodTur_Edited_Nicola.csv, and EfterkodDelure_Edited_Nicola.csv.

* Merge_Efterkod_DFs.ipynb: Merges these into the master dataset: Data/FastCheckTUData.csv.

* prep_fine_tuning.ipynb: Converts the master data into LLM-ready formats: full_200k_dataset_with_prompts.csv and train_challenger.jsonl.

### 3. The Experimental Pipelines
Approach A: Deterministic (Local)
Script: DET_OSRM.ipynb

* Process: Runs Levenshtein string-matching logic against the DAWA registry and performs local OSRM routing.

* Output: DET_final_corrected_df.csv

Approach B: Semantic / RAG (DTU HPC Cluster)
This phase requires high-performance compute resources and follows a multi-step sequence:

* Training: train.py uses train_challenger.jsonl to create the fine-tuned lora_challenger_model.

* Indexing: build_rag_index.py creates the FAISS vector store (address.index) from adresser_all_denmark.csv.

* Augmentation: augment_with_rag.py combines user prompts with retrieved context to create full_200k_dataset_RAG_READY.csv.

* Inference: inference.py runs the fine-tuned model to produce RAG_final_thesis_results.csv

* RAG_OSRM.ipynb: Performs OSRM distance/time recalculations on the RAG results.

* Output: Data/RAG_corrected_df.csv.

### 4. Final Validation & Psychophysical Analysis
Once the cluster outputs are retrieved, final routing is performed locally, and both methods are compared against the baseline.

* confidence_analysis_vs_baseline.ipynb: The final analysis hub. Compares DET and RAG outputs against the original data to generate:

The Unified Bias Matrix (Distance/Time error).

Taxonomy of Failure morphology.

Performance Visualizations (Venn diagrams and confidence distributions).

---
## ⚠️ Note on Data Privacy
The datasets used in this project (`TU Raw Data` and `Efterkod`) are the property of the Danish National Travel Survey and contain sensitive mobility information. Consequently, the `Data/` and `Addresses/` CSV files are excluded from this repository to comply with GDPR and DTU data privacy policies. Users wishing to replicate the study must provide their own authorized access to these data sources.
