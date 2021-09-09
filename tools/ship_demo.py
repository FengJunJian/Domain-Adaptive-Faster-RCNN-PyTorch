# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import json
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import evaluate_predictions_on_coco,prepare_for_coco_detection
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.data.datasets.evaluation import evaluate

from ship_predictor import COCODemo

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import os


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD33.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=600,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        # default=True,
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=['OUTPUT_DIR', '../log33'],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    iou_types = ("bbox",)
    model_dir = cfg.OUTPUT_DIR
    save_dir=os.path.join(model_dir, 'inference')
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if save_dir:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(save_dir, dataset_name)
            os.mkdir(output_folder)
            output_folders[idx] = output_folder

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        if os.path.exists(os.path.join(output_folder, 'predictions_demo.pth')):
            predictions=torch.load(os.path.join(output_folder, 'predictions_demo.pth'))
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
            #myeval(predictions, data_loader_val.dataset, output_folder, iou_type="bbox")
            evaluate(dataset = data_loader_val.dataset,
                     predictions=predictions,
                     output_folder=output_folder,
                     **extra_args)
            myeval(predictions, data_loader_val.dataset, output_folder, iou_type="bbox")

        else:
            pass
            # inference(
            #     model,
            #     data_loader_val,
            #     dataset_name=dataset_name,
            #     iou_types=iou_types,
            #     box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            #     device=cfg.MODEL.DEVICE,
            #     expected_results=cfg.TEST.EXPECTED_RESULTS,
            #     expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            #     output_folder=output_folder,
            # )
        if True:
            predictions = torch.load(os.path.join(output_folder, 'predictions.pth'))
            saveImg = os.path.join(output_folder, 'img')
            if not os.path.exists(saveImg):
                os.mkdir(saveImg)
            #visualization(predictions, data_loader_val.dataset, saveImg, )
        #synchronize()


    Video_flag = False
    if Video_flag:
        cam = cv2.VideoCapture(0)
        while True:
            start_time = time.time()
            ret_val, img = cam.read()
            composite = coco_demo.run_on_opencv_image(img)
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            cv2.imshow("COCO detections", composite)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
    else:
        img_root = 'E:/fjj/SeaShips_SMD/JPEGImages'
        img = cv2.imread(os.path.join(img_root, 'MVI_1474_VIS_00120.jpg'))
        start_time = time.time()
        composite,predictions = coco_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))

        cv2.imshow("COCO detections", composite)

        cv2.waitKey()
    cv2.destroyAllWindows()

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

if __name__ == "__main__":
    main()
