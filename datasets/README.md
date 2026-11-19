# Candidate datasets for the chronic disease models

Updated 2026-10-25: the notes below are the historical screening record of 22 October. Since then the v2 models have been trained on the CDC questionnaire and the Shanghai glucose data, and a separate heart disease status research model has been trained on UCI Cleveland; the description of the raw files has not been rewritten. Whole-table processing is described in the repository README.

Screening date: 22 October 2026. Six candidate groups were downloaded, keeping both the original archives and the extracted files; nothing was cleaned in place and no model was trained from these files at that point. Row counts, duplicate records and missing-value statistics come from inspecting the actual files and are recorded in `inspection.json`. The fact that a raw file can be downloaded does not mean its source quality and usage permission have been fully confirmed.

The data files, archives and inspection JSON referenced here are kept locally and are excluded from the published repository; only this note is shared.

## Preferred candidates

| Data | Actual scale | Main content and target | Suitability |
|---|---|---|---|
| Shanghai ShanghaiT1DM and ShanghaiT2DM | 12 type-1 patients and 100 type-2 patients; 16 and 109 record files | Continuous glucose with date and time, partial diet and medication records, plus clinical summary tables | Closer to continuous monitoring and future-glucose forecasting; cannot be used to train whether a healthy person will develop diabetes |
| CDC BRFSS 2015 diabetes health indicators | Full data 253680 rows with 21 input fields; also a balanced variant with 70692 rows | BMI, blood pressure history, lifestyle, self-rated health, predicting diabetes status classification | Source chain is comparatively clear and suits tabular classification; no raw continuous glucose, and age is group-coded |
| Kaggle Diabetes Prediction | 100000 rows, 8 input fields | Age, sex, BMI, hypertension history, heart disease history, smoking history, HbA1c, glucose; label `diabetes` | Fields are close to health indicator inputs; collection background and licence are not sufficiently documented, so it is not an unconditional first choice |
| Kaggle Cardiovascular Disease | 70000 rows, 11 input fields, plus id and label | Age, height and weight, systolic and diastolic pressure, cholesterol and glucose grades, lifestyle; label `cardio` | Blood-pressure-related inputs fit well; obvious outliers need handling and the licence is unconfirmed |

## Shanghai continuous glucose data

- Local folder: `shanghai_cgm/`. Figshare version 5 (2023-10-30) was downloaded and the archive MD5 was checked against the official metadata.
- Type 1: 12 patients, 16 record files, 15695 rows. Type 2: 100 patients, 109 record files, 112475 rows, of which 112462 rows have a non-empty CGM value.
- The main sampling interval is about 15 minutes, with missing measurements and irregular intervals. One patient may have several admission or visit records, so a record file must not be treated as a distinct patient.
- Actual record fields include `Date`, `CGM (mg / dl)`, `CBG`, `Blood Ketone`, `Dietary intake`, diet, and different medication administration forms. Diet and medication are sparse event fields; blank cells there must not all be treated as ordinary missing values.
- The summary tables contain age, sex, BMI, disease duration, complications, HbA1c and kidney function. They must be joined on patient and visit identifiers, and glucose and HbA1c use different units.
- A "past glucose window to future glucose" task can be constructed; high and low glucose alerts need their own target and thresholds. There is no healthy control group, so the data cannot be presented as future disease prediction for the general population.
- Publisher page: https://figshare.com/articles/dataset/diabetes_datasets_zip/21600933
- Collection page: https://figshare.com/collections/Diabetes_Datasets_ShanghaiT1DM_and_ShanghaiT2DM/6310860/2
- Original paper: https://doi.org/10.1038/s41597-023-01940-7
- Publisher licence: CC BY 4.0; keep the authors, the data version, the citation and a note of modifications when using it.

## CDC diabetes health indicators

- Local folder: `cdc_diabetes/`. The three files are different label or sampling variants of one source and cannot be treated as independent data for mutual validation.
- Three-class file: 0 = no diabetes or gestational diabetes only, 213703 rows; 1 = prediabetes, 4631 rows; 2 = diabetes, 35346 rows. The label meanings come from the publisher's description and the counts were verified.
- Important discrepancy: the publisher's text describes the positive class of the binary file as prediabetes or diabetes. Row-by-row comparison of the full three-class and binary files, after confirming that the 21 features and the row order match exactly, shows the actual mapping is 0 and 1 to 0, and 2 to 1, that is prediabetes is folded into the negative class. Do not copy the page description. The balanced variant has 35346 positives and 35346 negatives.
- The three-class file contains 23899 exactly identical rows; the full binary file contains 24206. Identical records in a limited discrete questionnaire do not automatically mean duplicate patients, and the data has no patient id; evaluation must consider the effect of identical feature vectors across splits.
- The raw data comes from the US CDC BRFSS 2015 survey; the Kaggle version is a curated dataset, not continuous device measurement. The publisher marks it CC0, but the original source and questionnaire definitions should be retained.
- Kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- UCI index: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

## Kaggle diabetes data, 100000 rows

- Local file: `diabetes_100k/diabetes_prediction_dataset.csv`.
- 8500 positives and 91500 negatives; 3854 exactly identical rows; the smoking field holds "No Info" for 35816 rows, so the absence of blanks does not mean the information is complete.
- No continuous time or patient identifier is provided. HbA1c and glucose directly reflect glucose metabolism, so a high classification score does not by itself prove predictive power for future disease.
- The official API licence states "Data files (c) Original Authors"; the description does not clearly provide the original hospital, study design or collection process, so it is not treated as verified clinical data or freely commercial data.
- https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

## Kaggle cardiovascular data, 70000 rows

- Local file: `cardiovascular_70k/cardio_train.csv`, semicolon separated.
- `cardio=0` for 35021 rows and `cardio=1` for 34979 rows; `age` is in days, `gluc` is a grade rather than a glucose concentration, and `id` cannot be used as a prediction feature.
- 1236 rows have systolic pressure not greater than diastolic pressure; systolic ranges from -150 to 16020 and diastolic from -70 to 11000, so obvious anomalies are present. 41 rows share identical features, which does not automatically mean duplicate patients.
- The page states the data was collected at a health examination, with no future follow-up outcome. The official API licence is Unknown; the collecting institution and the permitted scope still need verification.
- https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

## Small alternative datasets

- `early_diabetes/diabetes_data_upload.csv`: UCI Early Stage Diabetes Risk Prediction, 520 rows with 16 input fields, mostly a symptom questionnaire; 320 positive and 200 negative, 269 exactly identical rows and only 251 distinct complete rows. Suitable for a small pipeline demonstration, not as the main evidence of model quality. https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
- `heart_disease/processed.cleveland.data`: UCI Cleveland subset, 303 rows with 13 input fields and 1 label, containing 6 missing cells marked with a question mark. The original label runs from 0 to 4; when converted by convention to presence or absence of disease, there are 164 negatives and 139 positives. It needs clinical examination fields and is a classic tabular baseline candidate, not long-term monitoring data. https://archive.ics.uci.edu/dataset/45/heart+disease

## Current screening conclusion

The Shanghai data is worth keeping as the continuous monitoring forecasting candidate; the CDC data is worth keeping as the static health feature classification candidate. They serve different tasks and do not replace each other. The 100000-row diabetes data and the 70000-row cardiovascular data remain alternatives pending further verification of source and licence. None of the six groups was confirmed as the final training set at screening time; the v2 models were subsequently trained on the CDC questionnaire and the Shanghai glucose data as noted at the top.

`download_manifest.json` records download addresses, file lists and SHA-256 digests for the first five groups; `shanghai_article_metadata.json` records files, version and licence for the Shanghai data; each Kaggle publisher metadata file keeps the original description; `inspection.json` records the actual data inspection results.
