import sys
sys.path.append("./mmdetection")
sys.path.append('./AICITY2020_DMT_VehicleReID')
sys.path.append("/home/andrew/Documents/project/SmartTraffic/lab/ocr/ocr_pytorch_withLCX")

from vehicleInstance import vehicleInstance
from matching import matchingTwoVehicleLists,matchingLastSubImage
from AICITY2020_DMT_VehicleReID.demo import demoInference_withModel as getReIDFeature
from mmdet.apis import init_detector, inference_detector  # , show_result
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import os
import mmcv
import pycocotools.mask as maskUtils
import argparse
# from ocr import ocr

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

def parse_args():
    parser = argparse.ArgumentParser(description='Reid Pipeline for Smart Transportation Project')
    parser.add_argument('--input_folder', help='input_folder',
            default="./test_img", type=str)
    parser.add_argument('--output_folder', help='output_folder',
            default="./test_output", type=str)

    args = parser.parse_args()
    return args

'''
    main function for Vehicle Re-ID
'''
if __name__ == '__main__':
    args = parse_args()

    # mmdetection config
    # config_file = './mmdetection/configs/ms_rcnn/ms_rcnn_r50_fpn_1x_coco.py'
    config_file = "./mmdetection/configs/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco.py"
    # checkpoint_file = './mmdetection/weight/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth'
    checkpoint_file = "./mmdetection/weight/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth"
    model = init_detector(config_file, checkpoint_file,
                        device=torch.device("cuda:0"))

    # prepare ReId model
    config_file = "./AICITY2020_DMT_VehicleReID/configs/baseline_veri_r50.yml"
    cfg.merge_from_file(config_file)
    cfg.freeze()
    if config_file != "":
        # logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, 'r') as cf:
            config_str = "\n" + cf.read()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    num_classes = 576
    reid_model = make_model(cfg, num_class=num_classes).to('cuda')
    reid_model.eval()
    
    # root folder for images
    labeled_list = os.listdir(args.output_folder)
    to_label_list = os.listdir(args.input_folder)
    # clean same images
    for img in labeled_list:
        to_label_list.remove(img)

    mask_colors = [(0,0,255),(0,255,0),(255,0,0),(0,0,0),(155,155,0)]

    # start processing
    for img in tqdm(to_label_list):
        img_path = os.path.join(args.input_folder, img)
        ori_img = Image.open(img_path)
        (ori_width, ori_height) = ori_img.size
        output_img = np.zeros((ori_height,ori_width,3), dtype=np.uint8)

        # split ori_img to four sub-images
        
        half_width, half_height = int(ori_width/2), int(ori_height/2)
        frames = [ori_img.crop((0, 0, half_width, half_height)),
                ori_img.crop((half_width, 0, ori_width, half_height)),
                ori_img.crop((0, half_height, half_width, ori_height)),
                ori_img.crop((half_width, half_height, ori_width, ori_height))]
        
        # create empty vihicles list for each subImage
        subImage_0_vehicles = []
        subImage_1_vehicles = []
        subImage_2_vehicles = []
        subImage_3_vehicles = []
        all_vehicles = [subImage_0_vehicles, subImage_1_vehicles, \
            subImage_2_vehicles, subImage_3_vehicles]

        for subImage_id in [0, 1, 2, 3]:
            
            ori_img = frames[subImage_id]
            ori_img_arr = np.array(ori_img)
            # get bbox and seg result through mmdet
            result = inference_detector(model, ori_img_arr)
            bbox_result, segm_result = result

            # Add the segm_result together
            segms = []
            for i in range(80):
                segms += segm_result[0][i]
            segm_result = segms
            
            # Add the bbox_result together
            bboxes = np.vstack(bbox_result)

            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)]
            labels = np.concatenate(labels)

            crop_img_list = []
            bbox_list = []
            segms_list = []
            # parsing mmdet result
            for i in range(bboxes.shape[0]):
                label = labels[i]
                bbox = bboxes[i]

                # car=2, bus=5, truck=7
                # set confidence threshold to 0.6
                if not label in [2, 5, 7] or bbox[4] < 0.3:
                    continue

                # show cropped image
                crop_img = ori_img.crop(
                    (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))

                crop_img_list.append(crop_img)
                bbox_list.append(bbox)
                segms_list.append(segms[i])

            '''
            [Test Mode] Batchsize = 1 
            new_embedFeature_list = []
            for crop_img in crop_img_list:
                new_embedFeature_list.append((getReIDFeature([crop_img]))[0])
            '''
            if len(crop_img_list) == 0 or crop_img_list == []:
                continue
            new_embedFeature_list = getReIDFeature(crop_img_list, reid_model)

            for i in range(len(bbox_list)):
                # create new vehicle instance
                new_vehicle = vehicleInstance(subImage_id = subImage_id)
                new_vehicle.crop_image = crop_img_list[i]
                new_vehicle.mask = segms_list[i]
                new_vehicle.bbox = bbox_list[i]
                new_vehicle.embed_feature = new_embedFeature_list[i]

                all_vehicles[subImage_id].append(new_vehicle)

        
        '''
        Matching
        '''
        # pick up zoom-in image
        subImage_vehicleNum = [len(l) for l in all_vehicles]
        zoomIn_subImage_id = subImage_vehicleNum.index(min(subImage_vehicleNum))
        # print("==> zoomIn_subImage_id: ", zoomIn_subImage_id)
        
        # give id to instance in zoom-in frame
        for idx, vehicle in enumerate(all_vehicles[zoomIn_subImage_id]):    
            vehicle.reid_id = idx + 1

        # pick up biggest car in zoom-in frame as target vehicle
        '''
        target_vehicle = -1
        target_vehicle_square = -1
        for idx, vehicle in enumerate(all_vehicles[zoomIn_subImage_id]):
            bbox = vehicle.bbox
            bbox_square = abs(int(bbox[0])-int(bbox[1])) * abs(int(bbox[2])-int(bbox[3]))
            if bbox_square > target_vehicle_square:
                target_vehicle_square = bbox_square
                target_vehicle = idx
            
        all_vehicles[zoomIn_subImage_id][target_vehicle].reid_id = 1
        '''
        
        reference_id = zoomIn_subImage_id
        used = [reference_id]
        tmp_counter = 0
        matched_subImage = []
        for subImage_id in [0, 1, 2, 3]:
            if subImage_id in used:
                continue
            if len(matched_subImage) == 0:    
                matchingTwoVehicleLists(all_vehicles[reference_id], all_vehicles[subImage_id])
                reference_id = subImage_id
                used.append(subImage_id)
                matched_subImage.append(subImage_id)
            if len(matched_subImage) == 1:    
                matchingTwoVehicleLists(all_vehicles[reference_id], all_vehicles[subImage_id], calculateVector=True)
                reference_id = subImage_id
                used.append(subImage_id)
                matched_subImage.append(subImage_id)
            else: # process last subImage using the other two subImages
                [ref_id1, ref_id2] = matched_subImage
                matchingLastSubImage(all_vehicles[ref_id2], all_vehicles[subImage_id])

        # test
        '''
        for subImage_id in [0, 1, 2, 3]:
            ids = [vehicle.reid_id for vehicle in all_vehicles[subImage_id]]
            print(subImage_id, ":", ids)
        '''
            
        '''
        add color to segmentations
        '''
        finish_subImage = []
        for subImage_id in [0, 1, 2, 3]:
            ori_img = mmcv.imread(np.array(frames[subImage_id]))
            
            for vehicle in all_vehicles[subImage_id]:
                if vehicle.reid_id != 0:
                    color = mask_colors[vehicle.reid_id%5]
                    mask = vehicle.mask
                    ori_img[mask] = color * 1
                    
            mmcv_img = mmcv.image.imread(ori_img)
            finish_subImage.append(mmcv_img)
        
        # concat all subImage
        output_img[0:half_height, 0:half_width] = finish_subImage[0]
        output_img[0:half_height, half_width:]  = finish_subImage[1]    
        output_img[half_height:, 0:half_width]  = finish_subImage[2]
        output_img[half_height:, half_width:]  = finish_subImage[3]
        
        PIL_output_img = Image.fromarray(output_img)
        PIL_output_img.save(os.path.join(args.output_folder, img))
        
        # TODO : save all info to json/txt file
        # txtFile = img.split('.')[0] + ".txt"
        
        # only test one image
        # exit()


    #################################################################################################################################
    exit()
