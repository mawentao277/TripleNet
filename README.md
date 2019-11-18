# TripleNet: Triple Attention Network for Multi-Turn Response Selection in Retrieval-based Chatbots 
This repository contains resources of the following [CoNLL 2019](https://www.conll.org) paper.  

Title: TripleNet: Triple Attention Network for Multi-Turn Response Selection in Retrieval-based Chatbots   
Authors: Wentao Ma, Yiming Cui, Nan Shao, Su He, Wei-Nan Zhang, Ting Liu, Shijin Wang, Guoping Hu   
Link: [https://www.aclweb.org/anthology/K19-1069.pdf](https://www.aclweb.org/anthology/K19-1069.pdf)

## News
We have uploaded our source codes and put the dicts for the model in [google drive](https://drive.google.com/file/d/1wMYiowGHywX43EJebJaj0Pi2oEjbqcKX/view?usp=sharing).

## Notes
For reproducing the performance of TripleNet, please download the datasets of [Ubuntu](https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntudata.zip) and [Douban](https://github.com/MarkWuNLP/MultiTurnResponseSelection) and put them in the 'data' directory, then train or test the model just like the scripts in 'shell'. As we read the data via generator, so please shuffle the traning set before training.

## Requirements
Python3.6  
Keras2.2.4 (or >=2.0)  
Tensorflow1.10.0 (or >=1.10.0)  
(We run the codes in Python3.6 + Keras2.2.4 + Tensorflow1.10.0)  

## Citation
If you use the data or codes in this repository, please cite our paper
```
@inproceedings{ma-etal-2019-triplenet,
    title = "{T}riple{N}et: Triple Attention Network for Multi-Turn Response Selection in Retrieval-Based Chatbots",
    author = "Ma, Wentao  and
      Cui, Yiming  and
      Shao, Nan  and
      He, Su  and
      Zhang, Wei-Nan  and
      Liu, Ting  and
      Wang, Shijin  and
      Hu, Guoping",
    booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/K19-1069",
    pages = "737--746"
}

```

## Issues
If there is any problem, please submit a GitHub Issue.
