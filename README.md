An end-to-end LLM engineering pipeline demonstrating how to take a base foundational model (LLaMA-3.2-3B-Instruct), fine-tune it for a specialized domain (Mathematical Pedagogy), serialize/quantize the weights into a local production-grade GGUF engine, and rigorously stress-test its output reliability using multi-dimensional semantic and factual validation vectors.

Moving past ambiguous manual "vibe checks," this framework introduces an automated, hardware-aware evaluation loop that scores model responses on semantic alignment, high-temperature self-consistency, and external fact-grounding via live retrieval.

---

## Architecture & Pipeline Overview

[1. FINETUNING & MERGING]├── Base Model: Unsloth LLaMA-3.2-3B-Instruct (4-bit quantized)├── LoRA Fine-Tuning: Custom math tutoring dataset (Fine_Tune.ipynb)└── Merging & GGUF Quantization: Exported to rykyu-math-tutor.gguf (Q4_K_M)│▼[2. INFERENCE RUNTIME Engine]└── Served locally via Ollama Engine (ollama run rykyu-math-tutor)│▼[3. AUTOMATED RELIABILITY PIPELINE] (rykyu_reliability.py)├── Multi-Run Execution Loop (N=3, Temp=0.8) ──► Latency Telemetry├── Semantic Alignment Evaluation (all-MiniLM-L6-v2) ──► Cosine Similarity├── Stability / Diversity Check ──► Multi-path Centroid Variance└── Live Context Grounding Loop (DDG + Wikipedia) ──► RoBERTa-MNLI Entailment│▼[4. COMPREHENSIVE REACTION MATRIX]└── Aggregated Quality Report & Level-by-Level SLA Metrics Dashboards
---

## Key Components & Implementation Breakdown

### 1. Fine-Tuning & Quantization Pipeline (Fine_Tune.ipynb, Fine_Tune2.ipynb)
* Base Optimization: Leverages unsloth for 2x faster Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA) adapters[cite: 2].
* Weight Serialization: Merges raw LoRA adapter weights natively into foundational 16-bit safetensors.
* Quantization Setup: Converts weights via llama.cpp targeting the Q4_K_M GGUF format (1.91 GB footprint), optimizing memory constraints for fast local execution without compromising mathematical reasoning integrity.

### 2. Multi-Dimensional Reliability Framework (reliability.py)
The evaluation harness runs a 120-prompt dataset split across complex mathematical structural tiers (Level 1 to Level 3) using a verification pipeline:

* Semantic Similarity (Similarity): Maps generations into a dense vector space using sentence-transformers/all-MiniLM-L6-v2 to compute explicit cosine similarity against gold-standard reference rows[cite: 3].
* Generative Self-Consistency (Consistency): Executes 3 independent iterations (NUM_RUNS=3) per prompt under high-entropy sampling (TEMPERATURE=0.8)[cite: 3]. It measures the statistical drift across the generation paths to calculate variance penalties[cite: 3].
* Live Fact Grounding & Hallucination Penalty (H_Penalty): Programmatically queries DuckDuckGo Search (ddgs) and the Wikipedia API to fetch true domain facts[cite: 3]. A zero-shot Natural Language Inference (roberta-large-mnli) pipeline scores contradiction vs. entailment vectors between generated text and verified reference facts, enforcing an explicit H_Penalty step-down on the final score when discrepancies occur[cite: 3].
* Hardware Telemetry: Implements Apple Silicon/TensorFlow GPU memory growth constraints (tensorflow-metal) while compiling request-by-request performance latency SLAs[cite: 3].

