# VAV Course Project Report Outline

## 1. Research Question
- Our core question is: during fine-grained bird classification training, does a ViT gradually shift its attention toward more discriminative local regions?
- Course connection: we relate this process to concepts in cognitive science such as selective attention, global-to-local processing, and feature integration.

## 2. Motivation
- Practical value: fine-grained species recognition can support ecological monitoring, field observation, and automated classification assistance.
- Course value: we care not only about accuracy, but also about where the model "looks," how its attention patterns change, and whether these changes can be described using cognitive concepts.

## 3. Theoretical Foundations
- Dosovitskiy et al. (2021): the basic ViT architecture.
- Abnar and Zuidema (2020): attention rollout.
- Chefer et al. (2021): attention is not a complete explanation, so interpretation should remain cautious.
- Raghu et al. (2021): representational differences between ViTs and CNNs.
- Navon (1981): global precedence.
- Treisman and Gelade (1980): Feature Integration Theory.

## 4. Experimental Setup
- Dataset: CUB-200-2011.
- Model: a pre-trained `google/vit-base-patch16-224`, fine-tuned on CUB.
- Saved checkpoints: `epoch_000`, `epoch_001`, `epoch_003`, `epoch_005`, and `final`.
- Analysis targets:
  - rollout heatmap
  - all-layers average heatmap
  - CLS attention at selected layers
- Metrics:
  - validation accuracy
  - loss
  - attention entropy
  - top-k patch concentration
  - bbox overlap

## 5. Results to Include
- Training curves.
- Curves showing how attention metrics change across epochs.
- Attention overlays for the same image across different checkpoints.
- Success cases and failure cases.

## 6. Discussion
- Attention can be treated as a clue to how the model selects information.
- More concentrated attention after training does not mean the model truly has human attention mechanisms; it only means it shows a comparable pattern.
- We should connect the results to course concepts while also discussing the boundary between useful analogy and over-interpretation.

## 7. Limitations
- Attention is not a complete explanation.
- Bbox overlap is only a rough measure of whether the model focuses on key regions.
- Fine-grained classification results may still be influenced by dataset bias and background cues.
