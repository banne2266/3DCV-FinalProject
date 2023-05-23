### How to Run
    python mytrain.py
mytrain.py 會用我自己寫的mytrainer.py，跑時不用下指令，若有參數要改的地方可動config裡的yaml檔，裡面設的東西是對應到原作者的option.py (其中"name"對應到option裡的"model_name")

附註:
可修改config/train.yaml裡的Batch size(我設1)，
若影像是jpg則改config/train.yaml裡的"png"為 False

## Citation

    @article{zhang2022lite,
    title={Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation},
    author={Zhang, Ning and Nex, Francesco and Vosselman, George and Kerle, Norman},
    journal={arXiv preprint arXiv:2211.13202},
    year={2022}
    }
