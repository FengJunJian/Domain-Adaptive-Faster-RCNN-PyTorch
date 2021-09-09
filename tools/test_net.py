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
import time
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
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import evaluate_predictions_on_coco,prepare_for_coco_detection

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD12.yaml",# test_e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes_car
        # default= "configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=['MODEL.DOMAIN_ADAPTATION_ON',False,
                 'OUTPUT_DIR','log12'],
        # default= ['MODEL.DOMAIN_ADAPTATION_ON',False,
        #          'DATASETS.TEST',('ship_test_SeaShips_cocostyle',),'OUTPUT_DIR','logSMD2SS'],#'DATASETS.TEST',('visual_cocostyle',)
        #ship_test_SeaShips_cocostyle
        nargs=argparse.REMAINDER,type=str
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    print(args.opts)
    time.sleep(1)
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
    save_dir=os.path.join(model_dir,'inference')
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

            # for i in range(len(predictions)):
            #     predictions[i] = boxlist_nms(
            #         predictions[i], 0.1
            #     )  ####nms
            myeval(predictions, data_loader_val.dataset, output_folder, iou_type='bbox')
            # evaluate(dataset = data_loader_val.dataset,
            #          predictions=predictions,
            #          output_folder=output_folder,
            #          **extra_args)

        else:
            _ = checkpointer.load(cfg.MODEL.WEIGHT)
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
        if False:
            predictions = torch.load(os.path.join(output_folder, 'predictions.pth'))
            saveImgPath = os.path.join(output_folder, 'img')
            if not os.path.exists(saveImgPath):
                os.mkdir(saveImgPath)
            visualization(predictions, data_loader_val.dataset, saveImgPath, 0.95)
        synchronize()


def visualization(predictions,dataset,output_folder,threshold=0.5):#threshold=0.1,iou_type="bbox"
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

    def write_detection(im, dets,thiness=5,GT_color=None):
        for i in range(len(dets)):
            rectangle_tmp = im.copy()
            bbox = dets[i, :4].astype(np.int32)
            class_ind = int(dets[i, 4])
            if class_ind==7:
                continue
            # score = dets[i, -1]
            if GT_color:
                color=GT_color
            else:
                color=colors[class_ind]

            string = CLASS_NAMES[class_ind]
            # string = '%s' % (CLASSES[class_ind])
            fontFace = cv2.FONT_HERSHEY_COMPLEX
            fontScale = 1.5
            # thiness = 2

            text_size, baseline = cv2.getTextSize(string, fontFace, fontScale, thiness)
            text_origin = (bbox[0]-1, bbox[1])  # - text_size[1]
        ###########################################putText
            cv2.rectangle(rectangle_tmp, (text_origin[0] - 1, text_origin[1] + 1),
                               (text_origin[0] + text_size[0] + 1, text_origin[1] - text_size[1] - 2),
                               color, cv2.FILLED)
            cv2.addWeighted(im, 0.7, rectangle_tmp, 0.3, 0, im)
            im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thiness)
            im = cv2.putText(im, string, text_origin,
                             fontFace, fontScale, (0, 0, 0), thiness,lineType=-1)
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
        # if 'MVI_1474_VIS_00120' in img_info['file_name']:
        #     a=11111
        prediction = boxlist_nms(
            prediction, 0.5
        )####nms
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
        inds=np.where(dets[:,4]>threshold)[0]#label>0
        dets=dets[inds,:]
        inds=np.where(dets[:,5]>0)[0]#scores>threshold
        dets=dets[inds,:]
        im=write_detection(im,dets,thiness=2)
        #im=write_detection(im,gts,(0,0,255),thiness=2)
        #cv2.imshow('b',im)
        cv2.imwrite(os.path.join(output_folder,img_info['file_name']),im)


def myeval(predictions,dataset,output_folder,iou_type="bbox"):
    #from tools.voc_ap import voc_ap
    import pickle as plk
    from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import COCOResults
    #results = COCOResults(*iou_types)

    coco_boxes=prepare_for_coco_detection(predictions=predictions,dataset=dataset)
    file_path= os.path.join(output_folder, iou_type + ".json")
    with open(file_path, "w") as f:
        json.dump(coco_boxes, f)
    coco_dt = dataset.coco.loadRes(str(file_path)) if file_path else COCO()
    coco_gt=dataset.coco
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    p_a1 = coco_eval.eval['precision'][0, :, 0, 0, 2]  # (T, R, K, A, M)
    #r_a1 = coco_eval.eval['recall'][0, 0, 0, 2]  # (T, K, A, M)
    # pr_array2 = res.eval['precision'][2, :, 0, 0, 2]
    # pr_array3 = res.eval['precision'][4, :, 0, 0, 2]
    r_a1 = np.arange(0.0, 1.01, 0.01)
    #pr_c=[]
    pr_c={'total':coco_eval.eval}
    for catId in coco_gt.getCatIds():#各类AP
        coco_eval_c = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval_c.params.catIds = [catId]
        coco_eval_c.evaluate()
        coco_eval_c.accumulate()
        #coco_eval_c.summarize()
        #pr_c.append(coco_eval_c.eval)
        pr_c[catId]=coco_eval_c.eval

    if output_folder:
        with open(os.path.join(output_folder,"coco_PR_all.pkl"),'wb') as f:
            plk.dump(pr_c,f)
        # with open(os.path.join(output_folder,"coco_results.txt"),'w') as f:
        #     for k,v in results.results.items():
        #         if isinstance(v,dict):
        #             for k1,v1 in v.items():
        #                 f.write(str(k1)+'\t'+str(v1)+'\n')
        # for iou_type in iou_types:
        #     with open(os.path.join(output_folder,iou_type+"PR.txt"),'w') as f:
        #         for d1,d2 in zip(x,p_a1):
        #             f.write(str(d1)+'\t'+str(d2)+'\n')
    # pp=coco_eval_c.eval['precision'][0, :, 0, 0, 2]
    # rr = np.arange(0.0, 1.01, 0.01)
    # voc_ap(rr,pp,False)
    # T = len(p.iouThrs)
    # R = len(p.recThrs)
    # K = len(p.catIds) if p.useCats else 1
    # A = len(p.areaRng)
    # M = len(p.maxDets)
    # precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
    # recall = -np.ones((T, K, A, M))
    # scores = -np.ones((T, R, K, A, M))
    #T:10 iouThrs    - [.5:.05:.95]
    #R:101 recThrs    - [0:.01:1]
    #K:number of categories
    #A:4, object area ranges,[[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]->[all,small,medium,large]
    #M:3 thresholds on max detections per image, [1 10 100]
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    return coco_eval



if __name__ == "__main__":
    main()
