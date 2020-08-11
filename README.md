# Textfier: Text-Based Modifiers

[![Latest release](https://img.shields.io/github/release/gugarosa/textfier.svg)](https://github.com/gugarosa/textfier/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/textfier.svg)](https://github.com/gugarosa/textfier/issues)
[![License](https://img.shields.io/github/license/gugarosa/textfier.svg)](https://github.com/gugarosa/textfier/blob/master/LICENSE)

## Welcome to Textfier.

Dealing with text is not often a trivial task. Hence, this package provides a more straightforward interface to tackle text-based texts and modifications. Built on top of Huggingface's Transformers, Textfier is a wrapper focusing on the specific tasks we are currently researching.

Use Textifier if you need a library or wish to:

* Implement or use pre-defined tasks;
* Mix-and-match different approaches to solve problems;
* Because modifying text is fun.

Read the docs at [textfier.readthedocs.io](https://textfier.readthedocs.io).

Textfier is compatible with: **Python 3.6+**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.

---

## Getting started: 60 seconds with Textfier

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage, and follow the example. We have high-level examples for most tasks we could think of.

Alternatively, if you wish to learn even more, please take a minute:

Textfier is based on the following structure, and you should pay attention to its tree:

```yaml
- textfier
    - core
        - dataset
        - runner
        - task
    - stream
        - cleaner
        - tokenizer
    - tasks
        - language_modeling
        - named_entity_recognition
        - question_answering
        - seq2seq
        - sequence_classification
    - utils
        - loader
        - logging
        - metrics
```

### Core

The core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules.

### Stream

Every pipeline has its first step, right? The stream package serves as primary methods to clean and tokenize data.

### Tasks

Pre-defined tasks provide an easier framework when loading pre-trained models. Hence, this package serves as a wrapper around pre-trained models loading from Huggingface's Transformers.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use it as you wish than re-implementing the same thing over and over again.

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, textfier will be the one-to-go package that you will need, from the very first installation to the daily-tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever)!:

```bash
pip install textfier
```

Alternatively, if you prefer to install the bleeding-edge version, please clone this repository and use:

```bash
pip install .
```

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here, you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
