PW3 - Intelligent Decision Support Systems
Credit Decision Support System (Give Me Some Credit)

1. Project Structure
--------------------
- Documentation/ : Project report, architecture, use cases, Gantt and time sheet.
- Data/ (or data/ locally) : Raw CSV (GiveMeSomeCredit-training.csv) and processed splits.
- Models/ : Trained models and fitted explanation files (Models/explain/unusual_detector.json).
- Source/ : Python source code
    - Source/config.py : Paths and random seed (42).
    - Source/data/ : Preprocessing and train/val/test split script.
    - Source/explain/ : Explanations, similar cases search, unusual-case detector, stand-in models.
    - Source/app/ : CaseResult, orchestrator, use cases, and Streamlit app.
    - Source/tests/ : Pytest test suite.
- Demo/ : 3-5 minute video demo.
- Presentation/ : Presentation slides (max 15 slides).

2. Setup and Running
--------------------
1. Create a Python 3.13 virtual environment and install dependencies:
     pip install -r requirements.txt

2. Place the Kaggle dataset at:
     data/GiveMeSomeCredit-training.csv

3. Build the train/val/test splits and fit the unusual-case detector:
     python -m Source.data.build_dataset
     python -m Source.explain.unusual

4. Run the tests:
     pytest Source/tests

5. Launch the Streamlit application from the repo root:
     PYTHONPATH=. streamlit run Source/app/app.py
