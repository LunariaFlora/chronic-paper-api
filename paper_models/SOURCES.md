# Provenance of the adapted classifier

This directory holds the adapter for **one** model: the hypertension classifier published by the paper's authors. It is not a submission of original research.

## The published model

> **Ligang Ren, Jinhua Liang, Zenan Shi, Yinhao Li, Yurong Wang, Ying Xing, Xiangfu Meng, Bin Yang, Jing Cao.**
> *Machine Learning for Hypertension Classification After Blood Pressure Measurement: An Interpretable and Calibrated Approach.*
> **Expert Systems** (Wiley), published 2026-09-06. DOI: [10.1111/exsy.70419](https://doi.org/10.1111/exsy.70419)

The author's pipeline — median imputation, log1p, standardization, recursive feature elimination, a decision tree and isotonic calibration — comes from the `revision/` folder of the author-provided archive `hypertension-code-data.zip`. Only package import paths were changed; the mathematics is unchanged. The authors' `seed=42` selection and `selected_k=6` are kept, and the 424 held-out probabilities match the authors' saved reference values record by record.

The author's code package is **not redistributed** in this repository, and the trained weights are not either. Obtain them from the authors if you need to run the endpoint.

## What is deliberately absent

A second manuscript was shared with the same author group during the collaboration, but it is **not publicly published**. This repository therefore contains no description of its method, no implementation, no weights and no results. Unpublished research is not ours to describe in public, and its absence here is intentional rather than an omission.

## Data

The classifier was trained by its authors on a public hypertension dataset of 4240 records (positive prevalence 31.1 percent). The raw data is not distributed here; obtain it from the original publisher and follow that publisher's terms.

## Reporting numbers

Accuracy, recall and Brier score quoted in the repository are the authors' published figures, repeated for context. They were not produced or re-validated here. Nothing in this repository should be read as a claim about clinical performance.
