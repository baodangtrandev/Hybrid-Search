# Hybrid Search with Pinecone
## Overview
This repository demonstrates a hybrid search implementation using Pinecone. The hybrid search leverages both dense and sparse vector representations to enhance search results. This approach combines the strengths of dense embeddings (like those from neural networks) with sparse embeddings (such as those from traditional text representations) to improve search accuracy and relevance.
## Table of Contents
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Usage](#Usage)
- [Dataset](#Dataset)
- [Library](#Library)
- [Functions](#Functions)
  + [Hybrid Scale](#HybridScale)
  + [Hybrid Query](#HybridQuery)
- [Example](#Example)
- [Contributing](#Contributing)

## Requirements
- Python 3.6+
- Pinecone database [(Setup)](#SetupPineconeDatabase)
- Necessary Library

## Usage

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
## Installation
```bash
git clone https://github.com/baodangtrandev/Hybrid-Search
cd Hybrid-Search
```
If you don't have requirements library, try to install them. Easy to install necessary library:
```bash
pip install -r requirements.txt
```

