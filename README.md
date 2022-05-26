# TOP-TOKENS

## Usage
This computer program can generate topmost useful keywords from a multi-class corpus.

## How it works?
I have made use of Bayes' Theorem in order to find useful tokens for each of the classes (intents).

The main goal is to find:
```
P(token = t | intent = i)
```
..and this process is repeated for each of the intents and irrelevant tokens are dropped 
(which tokens `t`s have `P(token = t| intent = i) = 0` for some intent `i`).

## System Requirements
```
Ubuntu 20.04
Python 3.6
```

## Installation
```
python3.6 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## How to run?
```
- You have to create a folder called `intent-sample` under the root directory of the project.
- Put files containing samples.
- It assumes the files are in '.csv' format with no header and one file contains examples of one intent
  and each of the examples are newline separated.
- to run (after activating venv):
  python top-tokens.py
```
