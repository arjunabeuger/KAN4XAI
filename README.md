## KAN We Explain Better? Investigating Kolmogorov-Arnold Networks through a lens of XAI

Welcome to the repository for my research on Explainable AI (XAI) using Kolmogorov-Arnold Networks (KANs). This research aims to address the challenge of balancing performance and explainability in AI systems.

## Abstract

Explainable AI (XAI) is essential for making AI systems interpretable and transparent, especially in critical fields like medicine and finance. The trade-off between performance and explainability is a significant challenge, with uninterpretable deep-learning models often outperforming simpler, interpretable models. This research explores Kolmogorov-Arnold Networks (KANs) as an approach to address this issue. Unlike traditional Multi-Layer Perceptrons (MLPs), KANs are based on the Kolmogorov-Arnold representation theorem, making them inherently interpretable.

This research evaluates KANs' potential in three machine learning tasks: symbolic regression, symbolic classification, and as components within a larger architecture. The study compares KANs to established benchmarks, highlighting their ability to generate interpretable mathematical expressions and decision rules. Results demonstrate that KANs achieve competitive performance while offering enhanced transparency, suggesting their viability as a new paradigm in XAI. Key contributions include comparative performance- and interpretability analysis, efficacy in symbolic classification on imbalanced datasets, and integration within larger networks. This research underscores KANs' promise in balancing performance and explainability, crucial for broader AI adoption.

## Repository Structure

- `symbolic_regression/`: Contains code and data for symbolic regression tasks as well as integration task.
- `symbolic_classification/`: Contains code and data for symbolic classification task.
- `requirements.txt`: List of dependencies for the project.

## Getting Started

### Prerequisites

Make sure you have Python installed. You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
