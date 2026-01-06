# InfiniMetrics
An InfiniTensor-featured comprehensive accelerator evaluation framework

## Pre-commit Setup Guide

This project uses **pre-commit** to automatically enforce code style (Black) and linting (Flake8) before every commit.

### 1\. Installation

Install the pre-commit package via pip:

```bash
pip install pre-commit
```

### 2\. Setup (Run Once)

Run the following command in the project root directory to activate the git hooks:

```bash
pre-commit install
```

*Output should be: `pre-commit installed at .git/hooks/pre-commit`*

### 3\. Usage

  * **Automatic:** Just run `git commit` as usual.
      * If **Black** modifies your files: Run `git add .` again and re-commit.
      * If **Flake8** reports errors: Fix the errors, `git add`, and re-commit.
  * **Manual:** To check all files in the repository without committing:
    ```bash
    pre-commit run --all-files
    ```

### 4\. Skip Checks (Emergency Only)

If you need to bypass the checks for a specific commit:

```bash
git commit -m "your message" --no-verify
```
