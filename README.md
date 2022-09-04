Vietnamese Speech-to-Text :smile:
=====

# Introduction

In this project, I aimed to develop an end-to-end automatic speech recognition system in which I used diffenent frameworks such as Pytorch-lightning. Flask, Docker, Uwsgi, Nginx, and AWS service to deploy and extend the project into a simple production scenario. Also, I focused on Vietnamese language but other ones are easily modified and used. 

About the model, here I used several well-known ones(Deeespeech, Conformer CTC and RNN-Transducer). As usual, I'd love to write some notes about the models I used :raising_hand:. But now I don't have much time, so I update serveral days later...

# Setup

## Data preparation

The project used VIVOs, which is a widely-used Vietnamese public dataset. Download it and put it into `./local/data`:
```
└───VASR
    ├───local
    │   ├───data
    │   ├───ckpts
    │   ├───configs
    │   ├───outputs
    │   │   └───version_3        
    │   │       └───checkpoints  
    │   ├───src
    │   │   ├───augs
    │   │   ├───datasets
    │   │   ├───engine
    │   │   ├───losses
    │   │   ├───metrics
    │   │   ├───models
    │   │   │   ├───conformer
    │   │   │   │   └───base
    │   │   │   └───deepspeech
    │   │   ├───optimizers
    │   │   └───utils
    │   └───tools
    └───server
        ├───flask
        │   └───templates
        └───nginx
```

## Checkpoints

All the needed checkpoints can be found [here](Not now bro). You can download and put them into the project as the folder architecture above.

## EC2 service

First, if you are new to AWS, please create an account and access to the `EC2` service, then you launch a new instance and choose the instance type and suitable resource at the same time. Then create a new key pair for logging into the server you've just chosen. After all, you will see an new dashboard showing up and you shoud **copy** the `public DNS` then run on terminal:

```
chmod 400 speech_recognition.pem
scp -i speech_recognition.pem -r server/ ubuntu@<paste here>:~
```

Now you run: `ssh -i speech_recognition.pem ubuntu@<paste here>` to log into the server!

# Tools

Several tools are necessary to run the final web app, just follow the instruction:

## Denoise

Due to the cleanliness of the VIVOs dataset, it's hard to apply the trained model for noisy datasets and even real-life cases and therefore it needs to be added a denoising tool. I'm developing it which is seperately implemented with this project and then utilizing it through a public api ... Should be done soon. 

## Train and Evaluate

Modify the config file and simply just run:
```
export PYTHONPATH=/home/path_to_project/VASR/local
python3 tools/train.py
```

## Manage antifacts

Tensorboard was used here: 

```
tensorboard --logdir <path_to_log_folder> --load_fast true
```

## Web Demo

Finally, after you log into the server, you just need to run:

```
chomd +x init.sh
./init.sh
```

# Future works

Lots of works should be done later, it will take time haha:

- [ ] Develop denoising tool
- [ ] Enjoy the benefits of semi-supervised learning
- [ ] Build speech augmentation modules 
- [ ] Experience more approaches such as fine-tuning wav2vec based and 
- [ ] Develop better web user interface, new branch with streamlit

# References

Many thanks to the authors of the related papers and great implementations that I mainly based on:

- [Conformer Paper](https://arxiv.org/abs/2005.08100)
- [Deepspeech 2 Paper](https://arxiv.org/abs/1512.02595)
- [Transducer Paper](https://arxiv.org/abs/1910.12977)
- [Tuanio's implementation](https://github.com/tuanio/conformer-rnnt)
- [Openspeech's implementation](https://github.com/openspeech-team/openspeech)
- [Valerio Velardo](https://www.youtube.com/watch?v=ceNWWxjtG3U&list=PL-wATfeyAMNpCRQkKgtOZU_ykXc63oyzp&index=8&ab_channel=ValerioVelardo-TheSoundofAI)



