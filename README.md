## Table of contents
* [General info](#general-info)
* [Setup](#Setup)

## General info
A Convolutional Neural Network classify the different states “open” and “close” of a valve by using an acoustic sensor. 
The captured information is evaluated and provided in an OPC Unified Architecture using the [FreeOpcUa library](https://github.com/FreeOpcUa/python-opcua).

## Installation

**NOTE**: Python 3.6 or higher is required.

```bash
# clone the repo
$ git clone https://github.com/tlucky/cnn_audio.git

# change the working directory to cnn_audio
$ cd cnn_audio

# install python3 and python3-pip if they are not installed

# install the requirements
$ python3 -m pip install -r requirements.txt
```
