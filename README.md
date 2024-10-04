# AIDAO

## Setup
Install Docker.

sh
$ git clone https://github.com/VladimirKalajcidi/AIDAO.git
$ cd AIDAO/2/submision
$ docker build -t aidao .


## Usage
### Training and Inferencing
Execute following command in AIDAO directory.

sh
$ docker run -it -v "$(pwd):/app" aidao
root@hostname:/workspace# make build
root@hostname:/workspace# make train
root@hostname:/workspace# make run
