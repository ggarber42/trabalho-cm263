<!-- INTERNAL AUDIT — NOT FOR SUBMISSION. This is the /check-reporting working
report, not the official journal checklist. Do not upload to a submission portal. -->

# TRIPOD+AI Compliance Report (Metodologia + Resultados)

Manuscript: Previsão de faltas em consultas ambulatoriais de atenção secundária (GERCON Porto Alegre), CMP263, sections Metodologia and Resultados only
Target manuscript file: `entregaveis/metodologia.md` (source_sha256 `2f3fb907d075`) and `entregaveis/resultados.md` (source_sha256 `8f4110fb313a`)
Target version: v1 (first draft, 20/11/2026)
Guideline: TRIPOD+AI 2024 (Collins et al., BMJ 2024). TRIPOD 2015 is not applied in addition (TRIPOD+AI is a rewrite, not an add-on). Checklist text: `skills/check-reporting/references/checklists/TRIPOD_AI.md`.
Date: 2026-11-20
Assessed by: Claude (automated pre-screening). A methodologist should review before submission.

## Scope and applicability

- **Out of scope (not counted):** items 1, 1-AI, 2 (title, abstract), 3a, 3b (introduction), 19a, 19b, 20 (discussion), 22 (funding). These belong to sections written by the author and are not in these two files.
- **Applicability caveat:** TRIPOD+AI is written for prediction models of individual risk. The study compares forecasting models at the CNES-day level (aggregated counts) and one individual-level model (`paciente_hgb`, probability per appointment). The checklist is applied to the whole protocol; items are scored against both levels. Whether TRIPOD+AI is the right instrument for a count-series comparison is a decision for the author (see CLAUDE.md, seção 8).

## Part A: Summary

| Status | Count | Percentage (of 34 applicable items) |
|--------|-------|-------------------------------------|
| PRESENT | 13 | 38,2% |
| PARTIAL | 15 | 44,1% |
| MISSING | 6 | 17,6% |
| N/A | 5 | — |
| Out of scope | 9 | — |
| **Assessed total** | **39** | — |

Overall compliance: 13/34 (38,2%) PRESENT; 28/34 (82,4%) PRESENT or PARTIAL.

**Critical items:** the three non-waivable TRIPOD+AI items (input data and outcome timing; development and temporal validation; calibration reported) are present. Item 16 is partial because the calibration slope and intercept are not reported. The headline is **item 15-AI (model availability) is MISSING**: the TRIPOD+AI notes make it critical when the full model cannot be written as coefficients, which is the case for the gradient-boosting models. The other critical gaps are the participant characteristics table (13b) and the validation-period distribution comparison (13c), both MISSING.

## Part B: Item-by-item checklist

| # | Section | Item | Status | Location | Notes |
|---|---------|------|--------|----------|-------|
| 1 | Title/Abstract | Title identifies model development/validation, population, outcome | OUT OF SCOPE | — | Author's section |
| 1-AI | Title/Abstract | Title identifies AI/ML methods | OUT OF SCOPE | — | Author's section |
| 2 | Title/Abstract | Abstract summary | OUT OF SCOPE | — | Author's section |
| 3a | Introduction | Medical context and rationale, existing models | OUT OF SCOPE | — | Author's section |
| 3b | Introduction | Objectives (development, validation or both) | OUT OF SCOPE | — | Author's section |
| 4a | Methods, Source of data | Study design / data source, per development and validation sets | PRESENT | metodologia.md §2, §7 | Administrative scheduling records; training = years before each test year, test = 2023–2026 |
| 4b | Methods, Source of data | Key study dates (accrual start/end) | PRESENT | metodologia.md §3 | Data 2018 to 31/03/2026; test 2023–2026; no follow-up (outcome known on appointment date) |
| 4-AI | Methods, Source of data | Source of all input data types | PARTIAL | metodologia.md §2, §5 | Data are structured administrative fields and census/route variables; the text does not say explicitly that there is no imaging or free text. Add one sentence. |
| 5a | Methods, Participants | Setting, number and location of centres | PRESENT | metodologia.md §2, §3 | 111 municipal CNES (of 221) in Porto Alegre; coordinators and health districts available as predictors |
| 5b | Methods, Participants | Eligibility criteria | PARTIAL | metodologia.md §3 | Filters are listed, but they come from `reestruturacao.qmd` and were not re-run. Counts before the parquet filters are not available. |
| 5c | Methods, Participants | Treatments received | N/A | — | No treatment is assigned; the specialty is a predictor (metodologia §5) |
| 6a | Methods, Outcome | Outcome definition, how and when assessed | PRESENT | metodologia.md §4 | y = FALTANTE; known on the appointment date |
| 6b | Methods, Outcome | Actions to blind outcome assessment | N/A | — | Routine administrative outcome recorded by the service, not assessed by the investigators. A sentence saying so is advisable. |
| 7a | Methods, Predictors | Predictors defined, with timing | PRESENT | metodologia.md §5 | 30 candidate predictors listed in config.yaml; t0 = scheduling date; history uses only events before t0 |
| 7b | Methods, Predictors | Blinding of predictors | N/A | — | Routine data; predictors are fixed at scheduling. |
| 7-AI | Methods, Predictors | Preprocessing (normalisation, missing inputs, dimensionality reduction) | PARTIAL | metodologia.md §8 | Median imputation and standardisation are described for the MLP, GLMs and SARIMAX. Native categorical handling in the boosting models and the absence of dimensionality reduction are not stated. |
| 8 | Methods, Sample size | How the study size was arrived at | PARTIAL | metodologia.md §3, §7 | The study uses all available data for the period. No justification is given for the test length (T = 170 weeks). |
| 8-AI | Methods, Sample size | Effective sample size vs model complexity (EPV) | MISSING | — | No events-per-parameter figure for `paciente_hgb` (30 predictors) or for the series models. Splits file has n_treino (CNES-days) only. |
| 9 | Methods, Missing data | Handling of missing data, with method | PRESENT | metodologia.md §8 | Median imputation in the models that need it; boosting models accept NaN. Missing rate per variable (1,1% for `status` and distance variables) is in §2. |
| 10a | Methods, Statistical analysis | How predictors were handled | PRESENT | metodologia.md §5, §8 | Numeric and categorical lists; lags; no predictor selection |
| 10b | Methods, Statistical analysis | Model type, model-building, internal validation | PRESENT | metodologia.md §8 | Nested temporal validation (folds Y−2, Y−1); criteria WAPE and Brier; 24-configuration budget |
| 10c | Methods, Statistical analysis | How predictions were calculated (validation) | PRESENT | metodologia.md §8, §9 | Fitted on training data before the test year; NB predictive distribution with dispersion from inner folds |
| 10d | Methods, Statistical analysis | Performance measures | PRESENT | metodologia.md §9 | WAPE, bias, CRPS, coverage, width, ECE, AUC, Brier, log loss |
| 10e | Methods, Statistical analysis | Model updating (recalibration) | PRESENT | metodologia.md §8 | `hgb_poisson_k` (factor k over 28 days, clipped to [0,5; 2]) and k_h rescaling of the agenda, both from training data only |
| 10-AI-a | Methods, Statistical analysis | Architecture in sufficient detail to replicate | PARTIAL | metodologia.md §8 | Layer sizes of the MLP and the loss of each boosting model are given. The MLP activation (default ReLU) and output loss are not stated. |
| 10-AI-b | Methods, Statistical analysis | Training procedure (optimiser, learning rate, batch size, epochs, early stopping) | PARTIAL | metodologia.md §8 | No early stopping (stated). MLP optimiser, batch size and learning-rate schedule not stated (grid values only). |
| 10-AI-c | Methods, Statistical analysis | Software libraries with versions, hardware | PRESENT | metodologia.md §12 | `requirements.lock` with versions; CPU only, 16 processes. Python version differs between lock (3.12.3) and run (3.12.16); state it in the text. |
| 10-AI-d | Methods, Statistical analysis | Reproducibility (seeds, splits, code) | PRESENT | metodologia.md §12 | Master seed 20261116; deterministic splits; config.yaml; SHA-256 manifest. Code availability is item 21-AI. |
| 11 | Methods, Risk groups | Details on risk groups | N/A | — | No risk groups were created. Levels A/B are cohort strata; the overbooking quantile is a decision rule, not a group. |
| 12 | Methods, Development vs validation | Differences between development and validation data | N/A | — | No external validation; same source, setting, eligibility, outcome and predictors. Only the period differs (metodologia §7). |
| 13a | Results, Participants | Flow of participants, with and without outcome | PARTIAL | resultados.md §1, Tabela 1 | Figure 1 and cohort counts exist. Filter-level counts (A1–A4) are not available. Outcome counts in training and test are not reported in the text (only 32.099 faltas in the 8.073 CNES-days of the overbooking table). |
| 13b | Results, Participants | Characteristics of participants, with missing counts | MISSING | — | No descriptive table of predictors (demographics, history, distance) by year or level. `tabelas/ausentes_por_variavel_ano.csv` exists but is not in the Results. |
| 13c | Results, Participants | Validation: comparison of development vs validation distributions | MISSING | — | Not produced. Needed for the temporal validation (predictors and outcome by period). |
| 14a | Results, Model development | Number of participants and events in each analysis | PARTIAL | resultados.md §3, §9 | Test n (8.073 CNES-days) is given in §3. Training n and events per fold are not reported (see `tabelas/divisoes_municipal.csv`). |
| 14b | Results, Model development | Unadjusted association of each candidate predictor with the outcome | MISSING | — | Not reported. Required "if done"; state explicitly if not done. |
| 15a | Results, Model specification | Full model (coefficients and intercept) | PARTIAL | resultados.md §3; metodologia.md §8 | Not feasible for the gradient-boosting models; the text should say so. The GLM coefficients are not reported. |
| 15b | Results, Model specification | How to use the model | PARTIAL | metodologia.md §8; resultados.md §9 | The formula for μ (k_h × Σ probabilities) is in the Methods. The Results do not show how to apply it to a new appointment. |
| 15-AI | Results, Model specification | Model availability (repository, API, executable), input/output spec | MISSING | — | Final fitted objects are not in the package. Repository is local (no URL). **Critical per TRIPOD+AI notes.** |
| 16 | Results, Model performance | Performance with CIs; discrimination and calibration (slope and intercept) | PARTIAL | resultados.md §3, §4 | Discrimination and calibration (coverage, ECE, bias) are reported; CIs are given only for the gain vs baseline. Calibration slope and intercept are not reported. |
| 16-AI | Results, Fairness | Performance by demographic subgroups | PARTIAL | resultados.md §5 | Subgroups by level, year and the Materno Infantil stratum. Sex and ethnicity are not in the data (dicionario_variaveis.md). Age is available but not stratified. |
| 18 | Discussion, Limitations | Limitations (nonrepresentative sample, few events per predictor, missing data) | PARTIAL | resultados.md §12 | Covers small sample, calibration, horizon, missing premises, single city. Does not address events per predictor (see 8-AI). |
| 19-AI | Discussion, Interpretation | Potential sources of bias (data bias, label noise, distribution shift) | PARTIAL | resultados.md §12 | Single-city generalisation and data premises are listed. Distribution shift between periods and label noise (delay in recording outcomes) are not discussed. Full item belongs to the author's Discussion. |
| 21 | Other information | Supplementary resources (protocol, web calculator, data sets) | PARTIAL | metodologia.md §12 | Local repository and manifests. No DOI, URL or tag. |
| 21-AI | Other information | Trained model and code public availability, with URL or access instructions | MISSING | — | Not yet decided. Data cannot be shared (LGPD); code can be. |
| 22 | Other information | Funding and role of funders | OUT OF SCOPE | — | Author's section |

## Part C: Action items (priority order)

1. **[MISSING] Item 15-AI: model availability.** Required: state what is available (code in a repository with a URL or tag; trained objects if any; input and output specification for `paciente_hgb`). Suggested location: Results §3 (or a new Methods paragraph). Example text: "The code for all models, with the pinned versions and the configuration file, is available at [URL, tag vX]. The per-appointment model is not reported as coefficients; its input variables and output are described in Methods §5 and §8." `fixable_by_ai`: false (needs a repository decision from the author).
2. **[MISSING] Item 13b: participant characteristics.** Required: a table of predictors by test year and by level (count, mean or proportion, missing count). Suggested location: Results §2 (Tabela 1 extension). Source: `tabelas/ausentes_por_variavel_ano.csv` and the auditoria; the script that builds the table must be written. `fixable_by_ai`: true (after the table is generated).
3. **[MISSING] Item 13c: development vs validation comparison.** Required: predictors and outcome rate in training (years before each test) vs test. Suggested location: Results §2. `fixable_by_ai`: true.
4. **[MISSING] Item 8-AI: events per parameter.** Required: number of consultations and FALTANTE events in training for each test year, and events per candidate predictor (30 for `paciente_hgb`). Suggested location: Methods §3 or §8. Source: training split counts. `fixable_by_ai`: true.
5. **[PARTIAL] Item 16: calibration slope and intercept, and CIs.** Required: calibration slope and intercept (or the calibration-in-the-large) for `paciente_hgb` and the best series model, with CIs; CIs for WAPE and AUC. Suggested location: Results §4. `fixable_by_ai`: false until the bootstrap is run (approval needed for the long run).
6. **[MISSING] Item 14a/14b: training events, unadjusted associations.** Required: add training n and events per split to Results §3; state whether univariable associations were computed. `fixable_by_ai`: true for 14a; 14b needs a decision.
7. **[MISSING] Item 21-AI: code availability.** Required: a repository URL or access instruction, and a version tag for the article. `fixable_by_ai`: false.
8. **[PARTIAL] Items 10-AI-a and 10-AI-b: MLP details.** Required: activation, output loss, optimiser, batch size and learning-rate schedule of the MLP. Suggested location: Methods §8. Example text: "The MLP uses ReLU activation, squared-error loss, the Adam optimiser with the learning rate from the grid, and the scikit-learn default batch size (200); there is no early stopping." `fixable_by_ai`: true, but check the values against the code first.
9. **[PARTIAL] Item 4-AI: input data types.** Suggested sentence (Methods §2): "All inputs are structured administrative fields; no imaging or free text is used." `fixable_by_ai`: true.
10. **[PARTIAL] Item 5b: eligibility counts.** Required: counts before the parquet filters, or an explicit note that they are not reproducible from the repository (raw file absent). Suggested location: Results §1. `fixable_by_ai`: false (needs the raw file or a note).
11. **[PARTIAL] Item 7-AI: categorical handling and dimensionality.** Suggested sentence (Methods §8): "Categorical predictors are passed to the boosting models as native categories; no dimensionality reduction is applied." `fixable_by_ai`: true.
12. **[PARTIAL] Item 8: sample size justification.** Suggested sentence (Methods §3): "No formal sample size calculation was made. The test covers all available days of 2023 to 2026 (170 weekly observations for the Diebold-Mariano comparisons); no power calculation was performed." `fixable_by_ai`: true.
13. **[PARTIAL] Item 18: events per predictor in the limitations.** Add one limitation sentence once 8-AI is done. `fixable_by_ai`: true.
14. **[PARTIAL] Item 15a/15b: model specification and use.** Add a sentence to Results §3 saying that the gradient-boosting models have no coefficient form, and a short worked description of how μ is computed for a new appointment. `fixable_by_ai`: true.
15. **[PARTIAL] Item 16-AI: age stratification.** Age is available. Add a stratification by age band to Results §5, or state that it was not done. `fixable_by_ai`: true.
16. **[PARTIAL] Item 19-AI: distribution shift and label noise.** Add to Results §12: the 2026 bias (1,10) and the delay in recording outcomes as a possible source of label noise. `fixable_by_ai`: true.
17. **[PARTIAL] Item 21: supplementary resources.** Add the repository tag and the manifest file names once they are public. `fixable_by_ai`: false.
18. **[PARTIAL] Item 13a: flow.** Add the outcome counts per split to the Results text or Figure 1 caption. `fixable_by_ai`: true.
19. **[BOUNDARY] Results section.** Checked: one interpretive sentence (overbooking/mode explanation) was rewritten during this review and moved to Methods §5. No other boundary issue found in Results §1–§11. The limitations subsection (§12) is allowed to interpret.

## Part D: Machine-readable summary

```json
{
  "check_reporting_version": "1.1",
  "manuscript_title": "Previsão de faltas em consultas ambulatoriais de atenção secundária (GERCON Porto Alegre), Metodologia e Resultados",
  "target_manuscript": "entregaveis/metodologia.md; entregaveis/resultados.md",
  "target_version": "v1",
  "source_sha256": "metodologia.md 2f3fb907d075; resultados.md 8f4110fb313a",
  "guideline": "TRIPOD+AI",
  "guideline_version": "2024",
  "date": "2026-11-20",
  "total_items": 39,
  "out_of_scope": 9,
  "present": 13,
  "partial": 15,
  "missing": 6,
  "na": 5,
  "compliance_pct": 38.2,
  "present_or_partial_pct": 82.4,
  "critical_headline": "15-AI model availability MISSING (critical per TRIPOD+AI notes for non-presentable models)",
  "action_items": [
    {"item_number": "15-AI", "section": "Results", "item_name": "Model availability (AI)", "status": "MISSING", "suggested_location": "Results §3", "suggested_fix": "State repository URL and tag, input/output specification for paciente_hgb.", "fixable_by_ai": false},
    {"item_number": "13b", "section": "Results", "item_name": "Participant characteristics", "status": "MISSING", "suggested_location": "Results §2", "suggested_fix": "Descriptive table of predictors by test year and level, with missing counts.", "fixable_by_ai": true},
    {"item_number": "13c", "section": "Results", "item_name": "Development vs validation distributions", "status": "MISSING", "suggested_location": "Results §2", "suggested_fix": "Predictor and outcome distributions in training vs test period.", "fixable_by_ai": true},
    {"item_number": "8-AI", "section": "Methods", "item_name": "Effective sample size (EPV)", "status": "MISSING", "suggested_location": "Methods §3", "suggested_fix": "Report training consultations, FALTANTE events and events per predictor.", "fixable_by_ai": true},
    {"item_number": "14b", "section": "Results", "item_name": "Unadjusted association per predictor", "status": "MISSING", "suggested_location": "Results §2", "suggested_fix": "Report univariable associations, or state explicitly that they were not computed.", "fixable_by_ai": false},
    {"item_number": "21-AI", "section": "Other", "item_name": "Code and model availability", "status": "MISSING", "suggested_location": "Methods §12", "suggested_fix": "Repository URL or access instructions and article tag.", "fixable_by_ai": false},
    {"item_number": "16", "section": "Results", "item_name": "Performance with CIs; calibration slope/intercept", "status": "PARTIAL", "suggested_location": "Results §4", "suggested_fix": "Calibration slope and intercept with CIs for paciente_hgb and the best series model.", "fixable_by_ai": false},
    {"item_number": "14a", "section": "Results", "item_name": "Participants and events per analysis", "status": "PARTIAL", "suggested_location": "Results §3", "suggested_fix": "Add training n and events per split.", "fixable_by_ai": true},
    {"item_number": "10-AI-a", "section": "Methods", "item_name": "Model architecture detail", "status": "PARTIAL", "suggested_location": "Methods §8", "suggested_fix": "MLP activation and output loss.", "fixable_by_ai": true},
    {"item_number": "10-AI-b", "section": "Methods", "item_name": "Training procedure", "status": "PARTIAL", "suggested_location": "Methods §8", "suggested_fix": "MLP optimiser, batch size, learning-rate schedule.", "fixable_by_ai": true},
    {"item_number": "5b", "section": "Methods", "item_name": "Eligibility criteria", "status": "PARTIAL", "suggested_location": "Results §1", "suggested_fix": "Counts before filters, or note that they are not reproducible.", "fixable_by_ai": false},
    {"item_number": "4-AI", "section": "Methods", "item_name": "Input data types", "status": "PARTIAL", "suggested_location": "Methods §2", "suggested_fix": "State that all inputs are structured administrative fields.", "fixable_by_ai": true},
    {"item_number": "7-AI", "section": "Methods", "item_name": "Predictor preprocessing", "status": "PARTIAL", "suggested_location": "Methods §8", "suggested_fix": "Native categorical handling in boosting models; no dimensionality reduction.", "fixable_by_ai": true},
    {"item_number": "8", "section": "Methods", "item_name": "Sample size justification", "status": "PARTIAL", "suggested_location": "Methods §3", "suggested_fix": "State that no sample-size calculation was made and why.", "fixable_by_ai": true},
    {"item_number": "15a", "section": "Results", "item_name": "Full model specification", "status": "PARTIAL", "suggested_location": "Results §3", "suggested_fix": "State that boosting models have no coefficient form.", "fixable_by_ai": true},
    {"item_number": "15b", "section": "Results", "item_name": "How to use the model", "status": "PARTIAL", "suggested_location": "Results §3", "suggested_fix": "Worked description of computing mu for a new appointment.", "fixable_by_ai": true},
    {"item_number": "16-AI", "section": "Results", "item_name": "Subgroup performance (fairness)", "status": "PARTIAL", "suggested_location": "Results §5", "suggested_fix": "Add age-band stratification or state it was not done.", "fixable_by_ai": true},
    {"item_number": "18", "section": "Results", "item_name": "Limitations", "status": "PARTIAL", "suggested_location": "Results §12", "suggested_fix": "Add events-per-predictor limitation.", "fixable_by_ai": true},
    {"item_number": "19-AI", "section": "Results", "item_name": "Sources of bias", "status": "PARTIAL", "suggested_location": "Results §12", "suggested_fix": "Distribution shift (2026 bias) and outcome-recording delay.", "fixable_by_ai": true},
    {"item_number": "13a", "section": "Results", "item_name": "Participant flow", "status": "PARTIAL", "suggested_location": "Results §1", "suggested_fix": "Outcome counts per split in text or Figure 1 caption.", "fixable_by_ai": true},
    {"item_number": "21", "section": "Other", "item_name": "Supplementary resources", "status": "PARTIAL", "suggested_location": "Methods §12", "suggested_fix": "Repository tag and manifest names once public.", "fixable_by_ai": false}
  ]
}
```

**Field definitions:** `total_items` counts the 39 items assessed (9 out of scope are excluded). `compliance_pct` = PRESENT / (total_items − N/A) = 13 / 34 = 38,2%. `present_or_partial_pct` = (13 + 15) / 34 = 82,4%.
