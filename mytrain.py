from __future__ import absolute_import, division, print_function

#from options import LiteMonoOptions
from mytrainer import Trainer

#options = LiteMonoOptions()
#opts = options.parse()

import yaml


def ShowConfig(cfg):
    
    print("-------setting-----------")
    for k, v in cfg.items():
        if k != "dataset" and k !="ARCH" and k != "train_params":
            print(f"{k}: ", v)
    
    print("----dataset setting------------")
    dcfg = cfg["dataset"]
    for k, v in dcfg.items():    
        print(f"{k}: ", v)

    print("----model setting------------")
    acfg = cfg["ARCH"]
    for k, v in acfg.items():
        print(f"{k}: ", v)


    print("----training setting------------")
    tcfg = cfg["train_params"]
    for k, v in tcfg.items():
        print(f"{k}: ", v)

if __name__ == "__main__":

    cfg = None
    with open("config/train.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    ShowConfig(cfg)
    
    # cfg = 
    trainer = Trainer(cfg)
    trainer.train()
