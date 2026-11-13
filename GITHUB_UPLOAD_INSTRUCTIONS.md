# GitHub Upload Instructions

These steps are for you to do. Nothing has been uploaded for you, and no repository link exists yet.

## 1. Before you upload

1. Make sure `.env` does **not** exist in the folder, or is ignored (it is listed in `.gitignore`). Never upload API keys.
2. Run the tests: `python -m pytest -q`.
3. If you have an API key: put it in `.env`, run `python main.py` and `python main.py --demo`. This replaces the outputs in `outputs/` and `memory/` with real-LLM results, which is better for submission.
4. Read `outputs/final_lesson.md` yourself and fix anything you disagree with.

## 2. Create the repository on GitHub

1. Log in to github.com and click **New repository**.
2. Name it `rag-lesson-generator`.
3. Choose **Public** (or Private, if your assessors will be given access).
4. Do **not** tick "Add a README" (you already have one). Click **Create repository**.

## 3. Push from your computer

Run these commands inside the project folder. Replace `YOUR-USERNAME` with your GitHub username.

```bash
git init
git add .
git status                      # check that .env is NOT in the list
git commit -m "Self-evaluating lesson generator: agents, LangGraph workflow, memory, UI, tests"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/rag-lesson-generator.git
git push -u origin main
```

If Git asks for a password, GitHub needs a **personal access token** (Settings -> Developer settings -> Personal access tokens) or the GitHub CLI login (`gh auth login`).

## 3b. If you accidentally committed a key

Delete the key at the provider immediately and create a new one. Removing the file later does not remove it from Git history.

## 4. After upload

1. Open the repository page and check that the README shows correctly.
2. Check that the files from the folder structure in the README are present.
3. Copy the repository URL for your submission.
4. Optional: add a short description and topics (`langgraph`, `genai`, `rag`, `streamlit`).
