# AIDAO

## Setup
Install Docker.


```sh
$ git clone https://github.com/VladimirKalajcidi/AIDAO.git
$ cd AIDAO/2/submision
$ docker build -t aidao .
```

Add "model.pkl" file to directory
Add "predict.npy" file to data/ts_cut/HCPex


## Usage
### Training and Inferencing
Execute following command in AIDAO directory.

```sh
$ docker run -it -v "$(pwd):/app" aidao
root@hostname:/workspace# make build
root@hostname:/workspace# make train
root@hostname:/workspace# make run
```
