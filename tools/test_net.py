# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import json
import colorsys
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import evaluate_predictions_on_coco,prepare_for_coco_detection

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="./configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml",# test_e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes_car
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=['MODEL.WEIGHT','./','MODEL.DOMAIN_ADAPTATION_ON',False,
                 'DATASETS.TEST',('ship_test_SeaShips_cocostyle',),'OUTPUT_DIR','logSMD2SS'],#'DATASETS.TEST',('visual_cocostyle',)
        #ship_test_SeaShips_cocostyle
        nargs=argparse.REMAINDER,type=str
    )
    #basedir='../'
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    print(args.opts)
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # cfg.freeze()
    #filename=os.path.splitext(os.path.basename(args.config_file))[0]
    #label_flag=filename[-2]
    #unlabel_flag=filename[-1]

    model_dir = cfg.OUTPUT_DIR
    save_dir=os.path.join(model_dir,'test')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    #output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=model_dir)
    last_checkpoint=os.path.join(model_dir,'last_checkpoint')
    if os.path.exists(last_checkpoint):
        with open(last_checkpoint,'r') as f:
            lines=f.readlines()
            cfg.MODEL.WEIGHT=os.path.join(model_dir,os.path.basename(lines[-1]))
    else:
        cfg.MODEL.WEIGHT=os.path.join(model_dir,'model_final.pth')

    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if save_dir:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(save_dir, dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        if os.path.exists(os.path.join(output_folder, 'predictions.pth')):
            predictions=torch.load(os.path.join(output_folder, 'predictions.pth'))
            extra_args = dict(
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                iou_types=iou_types,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            )
            #data_loader_val.dataset.root

            evaluate(dataset = data_loader_val.dataset,
                     predictions=predictions,
                     output_folder=output_folder,
                     **extra_args)
            if True:
                saveImg=os.path.join(output_folder,'img')
                if not os.path.exists(saveImg):
                    os.mkdir(saveImg)
                visualization(predictions, data_loader_val.dataset, saveImg, )

        else:
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
            )
        synchronize()


def visualization(predictions,dataset,output_folder,):#threshold=0.1,iou_type="bbox"
    num_color = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
    hsv_tuples = [(x / num_color, 1., 1.)
                  for x in range(num_color)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    colors = [c[::-1] for c in colors]
    CLASS_NAMES=[None]*cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
    CLASS_NAMES[0]='__background__'
    for k,v in dataset.coco.cats.items():
        CLASS_NAMES[k]=v['name']

    def write_detection(im, dets,color=None,thiness=10):
        for i in range(len(dets)):
            bbox = dets[i, :4].astype(np.int32)
            class_ind = int(dets[i, 4])
            # score = dets[i, -1]
            if color is None:
                color=colors[class_ind]

            # if class_name:
            im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thiness)

            string = CLASS_NAMES[class_ind]
            # string = '%s' % (CLASSES[class_ind])
            fontFace = cv2.FONT_HERSHEY_COMPLEX
            fontScale = 2
            thiness = 2

            text_size, baseline = cv2.getTextSize(string, fontFace, fontScale, thiness)
            text_origin = (bbox[0], bbox[1])  # - text_size[1]
        ###########################################putText
        # im = cv2.rectangle(im, (text_origin[0] - 2, text_origin[1] + 1),
        #                    (text_origin[0] + text_size[0] + 1, text_origin[1] - text_size[1] - 2),
        #                    colors[class_ind], cv2.FILLED)
        # im = cv2.putText(im, string, text_origin,
        #                  fontFace, fontScale, (0, 0, 0), thiness)
        return im
    #coco_results = []
    Imgroot=dataset.root

    for image_id, prediction in enumerate(tqdm(predictions)):
        #print(image_id)

        if len(prediction) == 0:
            continue
        img_info = dataset.get_img_info(image_id)
        # original_id = dataset.id_to_img_map[image_id]
        original_id=img_info['id']
        image_width = img_info["width"]
        image_height = img_info["height"]
        # boxlist_for_class = boxlist_nms(
        #     prediction, 0.1
        # )
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xyxy")
        gts=[]
        for anns in dataset.coco.imgToAnns[original_id]:
            gttemp=anns['bbox']
            gt=[gttemp[0],gttemp[1],gttemp[0]+gttemp[2],gttemp[1]+gttemp[3]]
            gt.append(anns['category_id'])  # gt_label
            gts.append(gt)
        gts=np.array(gts)

        # gt=dataset.coco.imgToAnns[original_id]['bbox']
        # gt.append(dataset.coco.anns[original_id]['category_id'])  # gt_label
        # boxes = prediction.bbox.tolist()
        # scores = prediction.get_field("scores").tolist()
        # labels = prediction.get_field("labels").tolist()
        # mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        image_path=os.path.join(Imgroot,img_info['file_name'])
        im=cv2.imread(image_path)
        dets=prediction.bbox.numpy()
        dets = np.hstack([dets,np.reshape(prediction.get_field("labels").numpy(),[-1,1])])#labels
        dets = np.hstack([dets, np.reshape(prediction.get_field("scores").numpy(), [-1, 1])])  # scores
        inds=np.where(dets[:,4]>0.9)[0]#label>0
        dets=dets[inds,:]
        inds=np.where(dets[:,5]>0)[0]#scores>threshold
        dets=dets[inds,:]
        im=write_detection(im,dets,thiness=2)
        im=write_detection(im,gts,(0,0,255),thiness=2)
        #cv2.imshow('b',im)
        cv2.imwrite(os.path.join(output_folder,img_info['file_name']),im)



def myeval(predictions,dataset,output_folder,iou_type="bbox"):
    coco_boxes=prepare_for_coco_detection(predictions=predictions,dataset=dataset)
    file_path= os.path.join(output_folder, iou_type + ".json")
    with open(file_path, "w") as f:
        json.dump(coco_boxes, f)
    coco_dt = dataset.coco.loadRes(str(file_path)) if file_path else COCO()
    coco_eval = COCOeval(dataset.coco, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # with open(file_path, "w") as f:
    #     json.dump(coco_boxes, f)
    # res = evaluate_predictions_on_coco(
    #     dataset.coco, coco_boxes, file_path, iou_type
    # )



if __name__ == "__main__":
    main()
