# import sys
# sys.path.append('./AICITY2020_DMT_VehicleReID')

import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    '''
        
    config_file = "./configs/baseline_veri_r50.yml"
    cfg.merge_from_file(config_file)
    
    # print(args.opts)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    # logger.info(args)

    if config_file != "":
        # logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            # logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    num_classes = 576
    model = make_model(cfg, num_class=num_classes).to('cuda')
    model.eval()
    img_path_list = []

    img = Image.open("./data/VeRi/YongtaiPoint_Google.jpg")

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    img = val_transforms(img)
    img = torch.unsqueeze(img, 0)
    img_path_list.append(img)
    with torch.no_grad():
        img = img.to('cuda')
        feat = model(img)
        print("feat", feat.shape)


def demoInference(img_list):
    config_file = "./AICITY2020_DMT_VehicleReID/configs/baseline_veri_r50.yml"
    cfg.merge_from_file(config_file)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    # logger.info(args)

    if config_file != "":
        # logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            # logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    num_classes = 576
    model = make_model(cfg, num_class=num_classes).to('cuda')
    model.eval()
    img_path_list = []
    feat_list = []

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    for img in img_list:
        img = val_transforms(img)
        img_path_list.append(np.array(img))


    with torch.no_grad():
        img_path_list = torch.from_numpy(np.array(img_path_list)).to('cuda')
        feat_list = model(img_path_list)

    return feat_list


def demoInference_simple(img):
    config_file = "./AICITY2020_DMT_VehicleReID/configs/baseline_veri_r50.yml"
    cfg.merge_from_file(config_file)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    # logger.info(args)

    if config_file != "":
        # logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            # logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    num_classes = 576
    model = make_model(cfg, num_class=num_classes).to('cuda')
    model.eval()
    img_path_list = []

    # img = Image.open(imgPath)

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    img = val_transforms(img)
    img = torch.unsqueeze(img, 0)
    img_path_list.append(img)
    with torch.no_grad():
        img = img.to('cuda')
        feat = model(img)
        # print("feat", feat.shape)
        
        return feat
    
def demoInference_withModel(img_list, model):    
    img_path_list = []
    feat_list = []

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    for img in img_list:
        img = val_transforms(img)
        img_path_list.append(np.array(img))


    with torch.no_grad():
        img_path_list = torch.from_numpy(np.array(img_path_list)).to('cuda')
        feat_list = model(img_path_list)

    return feat_list