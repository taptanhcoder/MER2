<h1 align="center">Vietnamese Multimodal Emotion Recognition</h1>

<p align="center">
  <strong>Reliability-aware AI framework for Vietnamese emotion recognition from speech and text</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Research-Multimodal%20Emotion%20Recognition-purple.svg" alt="Research"></a>
  <a href="#"><img src="https://img.shields.io/badge/Language-Vietnamese-green.svg" alt="Vietnamese"></a>
  <a href="#"><img src="https://img.shields.io/badge/NLP-PhoBERT-blue.svg" alt="PhoBERT"></a>
  <a href="#"><img src="https://img.shields.io/badge/Speech-HuBERT-orange.svg" alt="HuBERT"></a>
  <a href="#"><img src="https://img.shields.io/badge/Fusion-Light--BiCA--Gate-ff69b4.svg" alt="Light-BiCA-Gate"></a>
  <a href="#"><img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch"></a>
</p>

<p align="center">
  <em>Vietnamese speech-text emotion recognition · Reliability-aware fusion · Cross-modal interaction · Class-wise adaptive gating</em>
</p>

---

## Abstract

This project presents a reliability-aware multimodal AI framework for Vietnamese emotion recognition from paired speech and text. Vietnamese emotion recognition remains challenging because multimodal resources are limited, emotional cues are heterogeneous, and speech and textual signals may have unequal reliability across utterances.

The proposed framework, **Length-aware Calibrated Light-BiCA-Gate**, combines a **PhoBERT-based text expert** and a **HuBERT-based speech expert** with temperature calibration, reliability feature extraction, compact bi-directional cross-modal interaction, and class-wise adaptive fusion. Instead of relying on a fixed fusion rule or simple confidence selection, the model dynamically balances text, speech, and interaction-level evidence for each emotion class.

The project is designed for five-class utterance-level emotion recognition:

<p align="center">
  <img src="https://img.shields.io/badge/Anger-😠-red?style=flat-square" alt="Anger">
  <img src="https://img.shields.io/badge/Fear-😨-purple?style=flat-square" alt="Fear">
  <img src="https://img.shields.io/badge/Happiness-😊-yellow?style=flat-square" alt="Happiness">
  <img src="https://img.shields.io/badge/Sadness-😢-blue?style=flat-square" alt="Sadness">
  <img src="https://img.shields.io/badge/Neutral-😐-lightgrey?style=flat-square" alt="Neutral">
</p>

---

## Architecture

<p align="center">
  <img src="report/Overall%20architecture%20of%20the%20proposed%20Length-aware%20Calibrated%20Light-BiCA-Gate%20framework.png" width="95%" alt="Overall architecture of the proposed Length-aware Calibrated Light-BiCA-Gate framework">
</p>

<p align="center">
  <strong>Length-aware Calibrated Light-BiCA-Gate</strong><br>
  PhoBERT text expert + HuBERT speech expert + calibrated reliability modeling + compact cross-modal interaction + class-wise fusion gate.
</p>

---

## Fusion Concept

<p align="center">
  <img src="https://img.shields.io/badge/Text%20Evidence-PhoBERT-blue?style=for-the-badge" alt="Text Evidence">
  <img src="https://img.shields.io/badge/Speech%20Evidence-HuBERT-orange?style=for-the-badge" alt="Speech Evidence">
  <img src="https://img.shields.io/badge/Interaction%20Evidence-BiCA-purple?style=for-the-badge" alt="Interaction Evidence">
  <img src="https://img.shields.io/badge/Fusion-Class--wise%20Gate-ff69b4?style=for-the-badge" alt="Class-wise Fusion">
</p>

The final prediction is produced by combining three sources of evidence:

| Evidence Source | Purpose |
|---|---|
| **Text Evidence** | Captures lexical and semantic emotion cues from Vietnamese transcripts |
| **Speech Evidence** | Captures acoustic and paralinguistic emotion cues from waveform signals |
| **Interaction Evidence** | Captures agreement or mismatch between what is said and how it is spoken |

This design allows the model to reason beyond unimodal predictions and use reliability-aware evidence allocation during fusion.

---

## Research Background

Emotion recognition is an important task in affective computing and human-centered AI. In multimodal emotion recognition, speech and text provide complementary signals: text captures semantic meaning, while speech carries acoustic cues such as pitch, intensity, rhythm, prosody, and voice quality.

For Vietnamese, the task is especially challenging because:

- Vietnamese multimodal emotion datasets are still limited.
- Speech and text may express emotion with different reliability.
- Generated transcripts can be short, ambiguous, or emotionally implicit.
- Speech signals may vary due to speaker characteristics and acoustic conditions.
- Vietnamese is a lexical-tone language, where pitch and voice quality are linked to both emotion and tone realization.

This project addresses these challenges through a calibrated and reliability-aware fusion strategy rather than a fixed or confidence-only fusion rule.

---

## Method Highlights

### PhoBERT Text Expert

The text branch uses PhoBERT to model Vietnamese transcripts and extract lexical-semantic emotion cues. A length-aware head-tail strategy is used to preserve informative content from both the beginning and ending parts of generated transcripts.

### HuBERT Speech Expert

The speech branch uses HuBERT to learn acoustic representations from waveform inputs. This branch captures paralinguistic cues that may not be explicitly expressed in text.

### Temperature Calibration

The framework applies temperature scaling to expert predictions before fusion. Calibrated probabilities are used to derive reliability indicators such as confidence, entropy, and top-two margin.

### Compact Bi-Directional Cross-Modal Interaction

The model uses compact token banks from both speech and text to model cross-modal relations in two directions:

- text attending to speech,
- speech attending to text.

This helps capture complementarity or disagreement between transcript content and vocal expression.

### Class-wise Adaptive Fusion Gate

The Light-BiCA-Gate module learns class-wise weights over text, speech, and interaction logits. This allows each emotion class to use different levels of evidence from each modality.

---

## Technology Stack

| Category | Technologies |
|---|---|
| **Programming** | Python |
| **Deep Learning** | PyTorch |
| **Vietnamese NLP** | PhoBERT, Hugging Face Transformers |
| **Speech Processing** | HuBERT, Torchaudio |
| **Multimodal Learning** | Cross-modal attention, adaptive fusion, reliability-aware gating |
| **Calibration** | Temperature scaling, confidence modeling |
| **Evaluation** | Macro-F1, Accuracy, Weighted-F1, UAR, confusion analysis |
| **Data Processing** | Pandas, NumPy, JSONL/CSV manifests |
| **Experiment Workflow** | YAML configs, multi-seed benchmarking, artifact export pipeline |
| **Research Engineering** | Reproducible training, feature export, fusion analysis, paper-ready reporting |

---

## Research Contributions

- Built a Vietnamese speech-text emotion recognition framework from paired audio and transcript inputs.
- Developed a PhoBERT-based text expert for Vietnamese transcript emotion modeling.
- Developed a HuBERT-based speech expert for acoustic emotion representation learning.
- Applied temperature calibration to improve the reliability of expert probabilities before fusion.
- Extracted reliability features to guide multimodal decision-making.
- Designed compact bi-directional cross-modal interaction between speech and text representations.
- Proposed Light-BiCA-Gate, a class-wise adaptive fusion module for reliability-aware multimodal emotion recognition.
- Conducted additional cross-benchmark analysis using MELD with Vietnamese-translated transcripts and original English audio.
- Completed an IEEE-style research paper describing the motivation, method, experiments, analysis, and limitations.

---

## Skills Demonstrated

<p align="center">
  <img src="https://img.shields.io/badge/NLP-Vietnamese%20Language%20Modeling-blue?style=flat-square" alt="Vietnamese NLP">
  <img src="https://img.shields.io/badge/Speech-Self--supervised%20Representation-orange?style=flat-square" alt="Speech">
  <img src="https://img.shields.io/badge/AI-Multimodal%20Fusion-purple?style=flat-square" alt="Fusion">
  <img src="https://img.shields.io/badge/ML-Model%20Calibration-green?style=flat-square" alt="Calibration">
  <img src="https://img.shields.io/badge/Research-Experiment%20Design-black?style=flat-square" alt="Research">
</p>

This project demonstrates practical research and engineering skills in:

- Vietnamese natural language processing,
- speech emotion recognition,
- multimodal deep learning,
- transformer-based text modeling,
- self-supervised speech representation learning,
- model calibration and reliability estimation,
- cross-modal interaction design,
- class-wise adaptive fusion,
- experimental evaluation and analysis,
- academic paper writing and research communication.

---

## Project Status

<p align="center">
  <img src="https://img.shields.io/badge/Core%20Framework-Completed-success?style=for-the-badge" alt="Core Framework">
  <img src="https://img.shields.io/badge/Research%20Paper-Completed-success?style=for-the-badge" alt="Research Paper">
  <img src="https://img.shields.io/badge/Repository-Finalized-blue?style=for-the-badge" alt="Repository">
</p>

The research pipeline is finalized around the proposed **Length-aware Calibrated Light-BiCA-Gate** framework.

Current status:

- Core framework completed
- Final fusion architecture selected
- Main Vietnamese evaluation completed
- Supplementary MELD evaluation completed
- Gate behavior analysis completed
- IEEE-style research paper completed
- Repository finalized as a research snapshot

---

## Paper

This repository accompanies the research paper:

> **Reliability-Aware Multimodal Fusion for Vietnamese Emotion Recognition from Speech and Text**

The paper presents the motivation, related work, dataset construction, proposed method, experimental setup, results, discussion, limitations, and future directions of the framework.

---

## Notes

- Dataset files are not redistributed in this repository.
- Experimental checkpoints and large output artifacts are not required for browsing the project.
- The MELD setting is used as a complementary benchmark analysis, not as a replacement for native Vietnamese speech-text evaluation.
- Exploratory fusion variants are kept as research utilities and are not part of the finalized paper framework.

---

## Citation

If this project is useful for your research or academic work, please cite the associated paper once available.

```bibtex
@misc{vietnamese_multimodal_emotion_recognition,
  title        = {Reliability-Aware Multimodal Fusion for Vietnamese Emotion Recognition from Speech and Text},
  author       = {To be updated},
  year         = {2026},
  note         = {Research code for Vietnamese speech-text multimodal emotion recognition}
}
