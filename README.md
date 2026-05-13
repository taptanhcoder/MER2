# Vietnamese Multimodal Emotion Recognition

<p align="center">
  <strong>Reliability-aware AI framework for Vietnamese emotion recognition from speech and text</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Research-Multimodal%20Emotion%20Recognition-purple.svg" alt="Research"></a>
  <a href="#"><img src="https://img.shields.io/badge/Language-Vietnamese-green.svg" alt="Vietnamese"></a>
  <a href="#"><img src="https://img.shields.io/badge/NLP-PhoBERT-blue.svg" alt="PhoBERT"></a>
  <a href="#"><img src="https://img.shields.io/badge/Speech-HuBERT-orange.svg" alt="HuBERT"></a>
  <a href="#"><img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch"></a>
</p>

<p align="center">
  <img src="report/Overall%20architecture%20of%20the%20proposed%20Length-aware%20Calibrated%20Light-BiCA-Gate%20framework.png" width="95%" alt="Overall architecture of the proposed Length-aware Calibrated Light-BiCA-Gate framework">
</p>

---

## Project Summary

**Vietnamese Multimodal Emotion Recognition** is a research-oriented AI project for recognizing emotions from paired **speech** and **text** inputs.

The project focuses on a reliability-aware multimodal fusion framework named:

> **Length-aware Calibrated Light-BiCA-Gate**

The framework combines a **PhoBERT-based text expert** and a **HuBERT-based speech expert**, then performs calibrated reliability modeling, compact cross-modal interaction, and class-wise adaptive fusion to predict utterance-level emotions.

The target emotion space includes:

```text
anger · fear · happiness · sadness · neutral
```
Research Motivation

Emotion recognition from Vietnamese speech and text remains challenging because:

Vietnamese multimodal emotion resources are limited.
Speech and text often carry different emotional cues.
One modality can be uncertain, noisy, or overconfident for a given utterance.
Simple late fusion or confidence-based selection may not fully capture modality reliability.
Vietnamese speech introduces additional complexity because pitch and voice quality are linked to both emotion and lexical tone.

This project explores a more robust fusion strategy that models how reliable each modality is before combining them.

Core Idea

Instead of directly averaging predictions from speech and text, the proposed framework learns to combine three evidence sources:

Evidence Source	Role
Text Evidence	Captures lexical and semantic emotion cues from Vietnamese transcripts
Speech Evidence	Captures acoustic and paralinguistic cues from the waveform
Interaction Evidence	Captures agreement or mismatch between what is said and how it is spoken

The final prediction is produced through a class-wise adaptive gate, allowing the model to balance text, speech, and interaction evidence for each emotion class.

Method Highlights
1. PhoBERT Text Expert

A Vietnamese language model is used to extract semantic and contextual information from transcripts.

Key design choices:

Vietnamese transcript modeling
Length-aware input handling
Head-tail truncation for long generated transcripts
Utterance-level text representation
2. HuBERT Speech Expert

A self-supervised speech model is used to represent emotional cues from audio.

Key design choices:

Waveform-based speech modeling
Acoustic-paralinguistic representation learning
Utterance-level speech embedding
Speech emotion classification head
3. Temperature Calibration

The framework applies calibration before fusion so that model confidence can be used more reliably.

Reliability signals include:

confidence
entropy
top-two margin

These features help the fusion module distinguish confident, uncertain, and ambiguous expert predictions.

4. Compact Bi-Directional Cross-Modal Interaction

The model does not treat speech and text as isolated branches only.

Instead, compact token banks from both modalities are used to model cross-modal interaction in two directions:

text attending to speech
speech attending to text

This helps capture complementary or conflicting evidence between transcript content and speech expression.

5. Light-BiCA-Gate Fusion

The final fusion module performs class-wise adaptive gating over:

text logits
speech logits
interaction logits

This design supports reliability-aware multimodal decision-making while remaining lightweight enough for small Vietnamese emotion datasets.

Technology Stack
Area	Tools / Technologies
Programming	Python
Deep Learning	PyTorch
NLP	PhoBERT, Hugging Face Transformers
Speech Processing	HuBERT, Torchaudio
Multimodal Learning	Cross-modal attention, adaptive fusion, reliability-aware gating
Calibration	Temperature scaling, confidence modeling
Evaluation	Macro-F1, Accuracy, Weighted-F1, UAR, confusion analysis
Experiment Management	YAML configs, multi-seed benchmarking, artifact export pipeline
Data Processing	Pandas, NumPy, JSONL/CSV manifests
Research Workflow	Reproducible training, calibration, feature export, fusion evaluation
Skills Demonstrated

This project demonstrates end-to-end research and engineering skills in:

Vietnamese NLP
speech emotion recognition
multimodal deep learning
transformer-based text modeling
self-supervised speech representation learning
model calibration
reliability-aware fusion
cross-modal interaction design
experiment reproducibility
benchmark preparation
academic paper development
research-oriented code organization
Dataset Setting

The main research setting is based on VNEMOS, a Vietnamese speech emotion dataset.

Because VNEMOS is originally speech-only, transcripts are generated and paired with the corresponding audio segments to create a speech-text emotion recognition setup.

The project also includes a supplementary evaluation setting using MELD, where:

English transcripts are translated into Vietnamese for the text branch.
Original English audio is preserved for the speech branch.
The setup is treated as a complementary benchmark analysis, not a replacement for native Vietnamese speech-text evaluation.
Research Contributions

This project contributes a complete research framework for Vietnamese multimodal emotion recognition:

Constructs a paired speech-text setup from a Vietnamese speech emotion dataset.
Builds separate PhoBERT and HuBERT emotion experts.
Applies temperature calibration before multimodal fusion.
Extracts reliability features to guide fusion decisions.
Introduces compact bi-directional speech-text interaction.
Uses class-wise adaptive gating for reliability-aware fusion.
Provides supplementary cross-benchmark evaluation with MELD.
Supports analysis of how text, speech, and interaction evidence contribute to final predictions.
Project Status

The research pipeline is finalized.

Current status:

Core framework completed
Final fusion architecture selected
VNEMOS evaluation completed
MELD supplementary evaluation completed
Gate behavior analysis completed
IEEE-style research paper completed
Repository finalized as a research snapshot
Paper

This repository accompanies the research paper:

Reliability-Aware Multimodal Fusion for Vietnamese Emotion Recognition from Speech and Text

The paper presents the motivation, dataset construction, proposed method, experimental setup, results, discussion, limitations, and future directions of the framework.

Notes
Dataset files are not redistributed in this repository.
Users should obtain datasets from their official sources.
Experimental outputs, checkpoints, and large artifacts are not required for browsing the project.
Exploratory fusion variants such as Bayes, XGBoost, and MoE-style gating are treated as experimental utilities and are not part of the final paper framework.
