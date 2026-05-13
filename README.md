# Vietnamese Multimodal Emotion Recognition

Reliability-aware multimodal emotion recognition framework for Vietnamese speech and text.

This repository contains the research codebase for Vietnamese utterance-level multimodal emotion recognition from paired speech and transcript inputs. The final framework combines a PhoBERT-based text expert and a HuBERT-based speech expert with temperature calibration, reliability feature extraction, compact bi-directional cross-modal interaction, and class-wise adaptive fusion through the proposed **Length-aware Calibrated Light-BiCA-Gate** model.

<p align="center">
  <img src="report/Overall%20architecture%20of%20the%20proposed%20Length-aware%20Calibrated%20Light-BiCA-Gate%20framework.png" width="95%" alt="Overall architecture of the proposed Length-aware Calibrated Light-BiCA-Gate framework">
</p>

## Overview

Emotion recognition from Vietnamese speech and text remains challenging due to limited multimodal resources, heterogeneous emotional cues, and the unequal reliability of speech and textual signals across utterances. Speech can carry acoustic-paralinguistic information such as prosody, intensity, rhythm, and voice quality, while text captures lexical and semantic cues. However, either modality can become unreliable depending on the utterance, speaker, transcript quality, or acoustic condition.

This project addresses these challenges through a reliability-aware multimodal fusion framework. Instead of using fixed late fusion or selecting the modality with the highest confidence, the proposed model estimates calibrated reliability signals and learns class-wise fusion weights over three evidence sources:

- text expert logits,
- speech expert logits,
- cross-modal interaction logits.

The framework is designed for five-class Vietnamese emotion recognition:

```text
anger, fear, happiness, sadness, neutral