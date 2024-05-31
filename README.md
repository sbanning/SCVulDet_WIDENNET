
WIDENNET is a python implementation of smart contract vulnerability detection using the Wide and Deep Neural Network.

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

4. `config\train_data\reent_contracts.txt`
* dataset for reentrancy vulnerability

5. `config\train_data\ts_contracts.txt`
* dataset for timestamp dependence vulnerability

## Introduction
WIDENNET is a project that aims to detect smart contracts with **reentrancy** and **timestamp dependency** vulnerabilities. Our methodology is based on the extension
of the Wide and Deep Neural Network in the area of smart contract vulnerability detection.

## Code Files

1. `SCVulDet_WIDENNET.py`
* this is the main and base class file. It is implemented in python. 

2. `config\model\Wide_Deep.py`
* WIDENNET class file.

3. `config\model_metrics.py`
* contains various metric visualization tools.

4. `opcode_vectorizer.py` 
* opcode vectorizer python file


## Running Project
* To run the program:
1. setup your environment using the packages in the requirements
2. in the arg_parser.py file, indicate which type of vulnerability by setting the appropriate '--vul_type' as the default value. (default='re_ent' implies you are testing for reentrancy vulnerability)
3. ensure you have the right dataset in place: reent_contracts.txt for reentrancy, ts_contracts.txt for timestamp dependence. 
4. execute SCVulDet_WIDENNET.py file