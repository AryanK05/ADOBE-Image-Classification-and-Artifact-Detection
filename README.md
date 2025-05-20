# ADOBE: Image Classification and Artifact Detection

Welcome to the repository for **ADOBE: Image Classification and Artifact Detection**—a project designed to tackle one of today's most pressing challenges in digital media: distinguishing real images from AI-generated ones, and making the process transparent, robust, and explainable.
This project was IIT Guwahati's official submission for Adobe Research's Mid-Prep Problem Statement at the Inter IIT Tech Meet 13.0

---
<p align="center">
  <img src="https://github.com/user-attachments/assets/c7aa4fd8-86f4-4839-bfbc-3e8262d41708" style="width: 80%; max-width: 800px;">
  <br>
  <img src="https://github.com/user-attachments/assets/d634ccad-e54f-4883-9d25-8dc5ae1f16e1" style="width: 80%; max-width: 800px;">
</p>

---

## 🚀 Project Overview

The explosion of AI image generation models like Stable Diffusion and GANs has brought both creative opportunity and new risks to the digital world. As these models become more advanced, the line between authentic and synthetic images blurs, fueling concerns around misinformation, digital forensics, and content authenticity.

This project provides a **lightweight, efficient, and interpretable solution** for:

- **Detecting AI-generated images** with high accuracy and low latency.
- **Explaining the artifacts** that reveal an image's synthetic origins, making the decision process transparent and trustworthy.

---

## 🧠 Key Features

- **Hybrid Model Architecture:**  
  Combines the local feature extraction power of **Convolutional Neural Networks (CNNs)** with the global context awareness of **Data-efficient Image Transformers (DeiT)**. This synergy delivers state-of-the-art performance while keeping the model size practical for real-world deployment.

- **Artifact Detection & Explanation:**  
  Goes beyond simple classification. By leveraging **vision-language models (VLMs)** and multimodal embeddings, the system not only flags AI-generated images but also pinpoints and explains artifacts (like unnatural textures or blending) in clear, human-readable language.

- **Robustness:**  
  The architecture is designed to withstand common adversarial attacks and image perturbations (e.g., noise, compression, geometric transformations), ensuring reliability in diverse scenarios.

- **Transparency:**  
  Every classification comes with an interpretable explanation, bridging the gap between black-box AI and human trust.

---

## 🏗️ How It Works

**Task 1: AI-Generated Image Detection**  
- **CNN** extracts fine-grained, local features (edges, textures, patterns).
- **DeiT** captures long-range, global dependencies and overall image structure.
- **Weighted Feature Fusion** combines both representations, allowing the model to dynamically balance local and global cues for optimal classification.
- **Output:** Real or AI-generated label.

**Task 2: Artifact Detection and Explanation**  
- Utilizes a **vision-language model (VLM)** to analyze images and generate concise explanations of detected artifacts.
- Employs a hierarchical, zero-shot classification pipeline using multimodal embeddings, so the system can generalize to new types of artifacts—even those not seen during training.
- **Output:** JSON files containing artifact classes and human-readable explanations for each image.

---

## Task-Specific Instructions

For detailed setup and execution steps, **please refer to the README files inside each task folder**.

### Task 1
- Two folders:  
  - `Super Model (Final Model)`  
  - `Experimentation`

### Task 2
- One folder containing the final code.

### Report and Presentation
- The Project Report and the Presentation given at the Inter IIT Tech Meet 13.0 at IIT Bombay can be found at the root of the repository. 


## 📊 Results Snapshot

- **Detection Accuracy:**  
  Hybrid CNN+DeiT model achieved >99% accuracy on our custom datasets and 97.8% on CIFAKE, outperforming baselines and demonstrating strong generalization to new generative models and adversarial perturbations.
- **Artifact Detection:**  
  The vision-language pipeline provides detailed, human-interpretable artifact descriptions, validated on both human- and GPT-annotated datasets.

---

## 📝 Project Report

For a deep dive into the methodology, experiments, and results, see the attached [project report](Image%20Classification%20and%20Artifact%20Detection%20Report.pdf).

---

## 🤝 Acknowledgements

This project is inspired by recent advancements in deep learning and vision-language modeling. We’d like to thank the research community for their valuable contributions, which laid the groundwork for this effort. For a detailed list of references and inspirations, please see the References section of the project report.

---

## 📬 Contact

Questions or feedback?  
Open an issue or reach out to any of the contributors via GitHub.

