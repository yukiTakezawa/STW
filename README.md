# STW distance
This is the demo code for the [Supervised Tree-Wasserstein distance](https://arxiv.org/abs/2101.11520).

## Requirements
Install requirements.
```
sudo pip install -r requirement.txt
```

## Dataset
The datasets used in the paper can be downloaded from [WMD](https://github.com/mkusner/wmd) and [S-WMD](https://github.com/gaohuang/S-WMD).

```
cd dataset
mkdir r8
python preprocessing_r8.py
```
`*_train.csv` is the training data, `*_eval.csv` is the validation data, and `*_test.csv` is the test data.
The data in `*_train_all.csv` is the combined data for training with the data for validation.

## Training
```
cd src
mkdir ../exp/r8
mkdir ../exp/r8/trial_3906
python main.py --data_path ../dataset/r8/r8_train.csv --eval_path ../dataset/r8/r8_eval.csv --save_path ../exp/r8/trial_3906 --epoch 30 --batch 100  --n_inner 3906 --multiply 5.0 --margin 10.0 
```
- `data_path` is the path of training data.
- `eval_path` is the path of validation data.
- `save_path` is the path where model parameter is saved. The parameter with the smallest loss in the validation data is saved as `best.pth` and the last parameter is saved as `latest.pth`.
- `n_inner` is the number of internal nodes. (The number of child node is set to 5.)
- `margin` is a margin value of contrastive loss.

## Evaluation
```
python save_batch_eval.py --train_path ../dataset/r8/r8_train_all.csv --test_path ../dataset/r8/r8_test.csv --model_path ../exp/r8/trial_3906/best.pth --n_inner 3906 --save_path ../exp/r8/trial_3906/best_test.pickle
python knn.py --eval_path ../exp/r8/trial_3906/best_test.pickle 
```
- `train_path` is the path of training data.
- `test_path` is the path of test data.
- `model_path` is the path in which the model parameters are saved.
- `save_path` is the path in which the results are saved.

## Citation
```
@inproceedings{takezawa2021supervised,
    title = {Supervised Tree-Wasserstein Distance},
    author = {Yuki Takezawa and Ryoma Sato and Makoto Yamada},
    booktitle = {International Conference on Machine Learning},
    year = {2021}
}
```