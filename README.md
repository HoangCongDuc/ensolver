# ensolver
This is the implementation for the paper [EnSolver: Uncertainty-Aware CAPTCHA Solver Using Deep Ensembles](https://arxiv.org/pdf/2307.15180.pdf).

## Installation
Follow [the official instruction](https://pytorch.org/get-started/previous-versions/#v1110) to install PyTorch 1.11.0.

Install `mmcv` version 1.5.3 using this command:
```
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

Install `mmdetection` version 2.25.0 by cloning the repository:
```
git clone -b v2.25.0 --single-branch git@github.com:open-mmlab/mmdetection.git
```
then install it:
```
cd mmdetection
pip install -e.
cd ..
```

## Generating data
The data generator is implemented in `captcha.py`. To see a demo for it, see `create_captcha.ipynb`. To generate the whole dataset and store it in COCO format, run:
```
python gen_data.py
```

## Training
To train the ensemble model, run the training script multiple times, each for a base model. To train a base model, run:
```
./dist_train.sh configs/yolox_m_captcha_config.py <number-of-gpus> --work-dir <working-directory> --seed <random-seed>
```
For example, to train the first base model, run:
```
./dist_train.sh config/yolox_m_captcha_config.py 4 --work-dir work_dirs/model0 --seed 0
```
Do remember to use different seed to train the base models!

After training, the models, use the file `collect_ckpt.py` to collect the best checkpoints from each training to a new directory. This makes the trained models easier to locate.
```
python collect_ckpt.py work_dirs models
```

## Inference and testing
To test the model, first unzip the public datasets from the file `testdata.zip`. You can also convert the COCO format dataset into the format of the datasets in this folder with
```
python coco2text.py data/coco/annotations/test.json data/coco/images/test testdata/generated
``` 
Then, you can run this to get the predictions from the models:
```
python test_model.py testdata/generated config/yolox_m_captcha_config.py --ckpt-path models/model0.pt
h --out-path generated.txt --batch-size 32
```