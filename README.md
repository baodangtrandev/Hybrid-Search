# Hybrid Search with Pinecone
## Overview
This repository demonstrates a hybrid search implementation using Pinecone. The hybrid search leverages both dense and sparse vector representations to enhance search results. This approach combines the strengths of dense embeddings (like those from neural networks) with sparse embeddings (such as those from traditional text representations) to improve search accuracy and relevance.
## Table of Contents
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Dataset](#Dataset)
- [Library](#Library)
- [Example](#Example)
- [Contributing](#Contributing)

  ## Installation
```bash
git clone https://github.com/baodangtrandev/Hybrid-Search
cd Hybrid-Search
```
If you don't have requirements library, try to install them. Easy to install necessary library:
```bash
pip install -r requirements.txt
```
If you don't have jupyter notebook, please install:
```bash
pip install --user jupyterlab
```
Connect to Jupyter Lab:
```bash
jupyter notebook
```

## Requirements
- Python 3.6+
- Pinecone database [(Setup)](#Setup-Pinecone-Database)
- Necessary Library
- Jupyter NoteBook

## Dataset
I use [pubmed_qa](https://huggingface.co/datasets/qiaojin/PubMedQA) dataset on on Hugging Face Datasets. I download it like so:
```bash
!pip install datasets
from datasets import load_dataset
pubmed = load_dataset(
   'pubmed_qa',
   'pqa_labeled',
   split='train'
)
```
## Library
You need install some library to run model:
 - [Dataset Hugging Face library](https://github.com/huggingface/datasets)
 - [Transformers library](https://github.com/huggingface/transformers)
 - [rank-bm25 library](https://github.com/dorianbrown/rank_bm25)
 - [Sentence Transformers library](https://github.com/UKPLab/sentence-transformers)
 - [Pinecone Client library](https://github.com/pinecone-io/pinecone-python-client)
 - [Tqdm library](https://github.com/tqdm/tqdm)

## Setup Pinecone Database


