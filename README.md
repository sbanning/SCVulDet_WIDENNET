# SCVulDet_WIDENNET
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
* **Reentrancy**: config > train_data > reentrancy_sample
* **Timestamp**: config > train_data > timestamp_sample
* **processed opcode vectors**: config > train_data > contracts_opcodes_vectors

## Introduction
WIDENNET is a project that aims to detect smart contracts with **reentrancy** 
and **timestamp dependency** vulnerabilities. Our methodology is based on the extension
of the Wide and Deep Neural Network in the area of smart contract 
vulnerability detection.

## Code Files

1. `SCVulDet_WIDENNET.py`
* main and base class file.

2. `config\model\Wide_Deep.py`
* WIDENNET class file.

3. `config\model_metrics.py`
* contains various metric visualization tools.

##Running Project
* To run the program, execute SCVulDet_WIDENNET.py file
