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

## 📂 Project Structure
* `/experiments/deterministic`: Scripts for Levenshtein matching and DAWA registry queries.
* `/experiments/semantic`: RAG implementation, FAISS index construction, and local LLM prompting.
* `/data_baseline`: Documentation on the "Efterkod" (Post-code) stratum used as the experimental ground truth.
* `/results`: Notebooks generating the error matrices and taxonomy of failure.

---
*This research serves as a proof-of-concept for the "Golden Cascade" architecture—proposing that future survey systems should use deterministic filters for volume and semantic models for the complex residual tail.*
