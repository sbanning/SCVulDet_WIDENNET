## Introduction
WIDENNET is a python implementation of smart contract vulnerability detection using the Wide and Deep Neural Network. The goal is to detect smart contracts with **reentrancy** and **timestamp dependency** vulnerabilities. Our methodology is based on the extension of the Wide and Deep Neural Network in the area of smart contract vulnerability detection.

## Requirements

#### The following packages are required to run WIDENNET
* **python**3.0
* **Tensorflow** 2.9.0
* **sklearn**
* **matplotlib**
* **gensim**
* **solcx**
* **pyevmasm**
* **pandas**

## Dataset
We utilized a publicly available smart contract dataset from GitHub, published by the authors of a notable research paper in blockchain security and smart contract analysis. This dataset comprises a comprehensive collection of smart contracts sourced from the Ethereum Platform (over 96%), GitHub repositories, and
blog posts that analyze contracts. The results of our work were compared against the performance metrics published in the same paper that provided
the dataset on GitHub. Link to the dataset: https://github.com/Messi-Q/Smart-Contract-Dataset

 * dataset for reentrancy vulnerability
 `config\train_data\reent_contracts.txt`


* dataset for timestamp dependence vulnerability
`config\train_data\ts_contracts.txt`


## Code Files

* this is the main and base class file. It is implemented in python.
`SCVulDet_WIDENNET.py` 

* WIDENNET class file.
  `config\model\Wide_Deep.py`

* contains various metric visualization tools.
  `config\model_metrics.py`

* opcode vectorizer python file
  `opcode_vectorizer.py` 



## Running Project
* To test WIDENNET:
1. setup your environment using the packages in the requirements
2. ensure you have the right dataset in place: `reent_contracts.txt` for reentrancy, `ts_contracts.txt` for timestamp dependence
3. for vulnerability type: `ts` for timestamp and `re` for reentrancy
   
4. For timestamp dependence, execute the command:
```
  python3 SCVulDet_WIDENNET.py .\ts_contracts.txt ts
```
   
4. For reentrancy, execute the command:
```
  python3 SCVulDet_WIDENNET.py .\reent_contracts.txt re
```
   
