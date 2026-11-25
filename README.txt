INTELLIGENT DECISION SUPPORT SYSTEMS — PRACTICAL WORK 3 (PW3, TEAMWORK)
Retail Credit Scoring IDSS ("Give Me Some Credit")
========================================================================

1. STRUCTURE AND CONTENTS OF THE DELIVERABLE ZIP
------------------------------------------------------------------------
This archive follows the exact folder structure specified in PracticalWork3IDSS.pdf:

  1. Documentation/
     Contains the project report covering sections 1.a through 1.m:
     - Domain of application & main supported decisions (Approve / Review / Reject)
     - Functional architecture of the hybrid IDSS (data-driven + knowledge-based)
     - Data pre-processing summary, flowchart, post-processing and validation
     - Model-driven / knowledge-based techniques (policy rules, cost-ratio policy, OOD check)
     - Testing use cases (Cases 1 to 4), global evaluation, conclusions & future work
     - Gantt diagram, tasks assignment (Persons A-E), and team time sheet

  2. Data/  (git-ignored locally as data/ or Data/)
     - GiveMeSomeCredit-training.csv          : Original raw dataset (150,000 borrowers)
     - processed/train_readable.csv           : Preprocessed training split (96,000 rows, readable units)
     - processed/val_readable.csv             : Preprocessed validation split (24,000 rows, readable units)
     - processed/test_readable.csv            : Preprocessed test split (30,000 rows, readable units)
     - processed/train_model.csv              : Log-transformed & standardised training split
     - processed/val_model.csv                : Log-transformed & standardised validation split
     - processed/test_model.csv               : Log-transformed & standardised test split
     - processed/preprocessing.joblib         : Fitted Preprocessor artefact for runtime inference

  3. Models/
     Trained classifiers and readable model summaries:
     - logistic_regression.joblib             : Fitted Logistic Regression classifier
     - logistic_regression_weights.json       : Readable standardized weights & odds multipliers
     - decision_tree.joblib                   : Fitted Decision Tree classifier
     - xgboost.joblib                         : Fitted XGBoost classifier
     - xgboost_feature_importance.json        : Readable XGBoost feature importances
     - ensemble_config.json                   : Ensemble weights (5% LR, 5% Tree, 90% XGBoost) & features
     - explain/unusual_detector.json          : Fitted out-of-distribution thresholds and training bounds

  4. Source/
     Complete Python source code and training/evaluation scripts:
     - config.py                              : Shared paths and random seed (RANDOM_STATE = 42)
     - train_models.py                        : Training script for the 3 models and ensemble config
     - model_loader.py                        : Loader for the saved classifiers and ensemble config
     - export_readable_models.py              : Exports readable rules/weights/importances into Models/
     - build_deliverable_zip.py               : Packages the final PW3 ZIP archive
     - data/
         preprocessing.py                     : Cleaning rules, income imputer, derived ratios, scaler
         build_dataset.py                     : Builds the stratified 96k / 24k / 30k data splits
     - explain/
         types.py, features.py, pipeline.py   : Shared explanation types, field metadata, RiskPipeline
         factors.py                           : Exact per-input SHAP attributions merged across models
         advice.py                            : Actionable counterfactual ("what-if") search
         similar.py                           : Nearest-neighbour search over past training applicants
         unusual.py                           : Out-of-distribution / unusual applicant detector
         case.py                              : Unified explain_case() entry point
     - decision/
         policy.py, rules.py, engine.py       : Cost-ratio risk slider, bank policy rules, DecisionEngine
         service.py, evaluate.py              : App entry point evaluate_decision() & validation sweep
         narrative/                           : Guarded LLM (Ollama / OpenAI) + template text generator
     - app/
         case_result.py, use_cases.py         : Unified CaseResult contract & 4 testing scenarios
         orchestrator.py                      : End-to-end IDSSOrchestrator connecting parts A, B, C, D
         app.py                               : Streamlit UI (Officer view, Customer view, Test cases, Models & rules)
     - tests/                                 : Automated pytest test suite (116+ tests)
     - notebooks/ (in ZIP; root locally)      : EDA.ipynb and model_training.ipynb

  5. Demo/
     Contains the 3-5 minute video demonstration of the IDSS prototype.

  6. Presentation/
     Contains the public presentation slides (up to 15 slides).

  7. README.txt
     This file describing the structure, contents, and execution instructions.


2. HOW TO REPRODUCE, TEST, AND RUN THE SYSTEM
------------------------------------------------------------------------
Run all commands from the root directory with Python 3.13:

1. Install dependencies:
     pip install -r requirements.txt

2. Place the raw dataset at Data/GiveMeSomeCredit-training.csv (or data/GiveMeSomeCredit-training.csv)
   and build the preprocessed splits:
     python -m Source.data.build_dataset

3. (Optional) Re-train models, re-fit the unusual-case detector, and export readable model summaries:
     python -m Source.train_models
     python -m Source.explain.unusual
     python -m Source.export_readable_models

4. Run the automated test suite:
     pytest Source/tests

5. Launch the Streamlit user interface:
     PYTHONPATH=. streamlit run Source/app/app.py

   (Optional LLM configuration for live narrative generation instead of deterministic templates:
    set IDSS_LLM_BACKEND=ollama with Ollama running locally, or IDSS_LLM_BACKEND=openai with OPENAI_API_KEY.)

6. Package the final deliverable ZIP:
     python -m Source.build_deliverable_zip --year 2026-2027 --team <TeamNumber> --title <TitleOfWork>
