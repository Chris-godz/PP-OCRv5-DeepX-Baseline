import os
import sys
import time
import queue
import threading

from dx_engine import InferenceEngine as IE

from .utils import get_rotate_crop_image, torch_to_numpy, filter_tag_det_res
from .utils import rec_router, split_bbox_for_recognition, merge_recognition_results

from typing import List
import numpy as np
import torch
import cv2

engine_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(engine_path)

from baidu import det_preprocess, cls_preprocess, rec_preprocess
from models.ocr_postprocess import DetPostProcess, ClsPostProcess, RecLabelDecode
from preprocessing import PreProcessingCompose

class Node():
    def __init__(self):
        pass
    
    def prepare_input(self, inp: np.ndarray) -> np.ndarray:
        if inp.ndim == 3:
            inp = np.expand_dims(inp, 0)
        if isinstance(inp, np.ndarray):
            inp = torch.as_tensor(inp)
        inp = inp.permute(0,2,3,1).contiguous()
        return torch_to_numpy(inp)
    
    def thread_postprocess(self):
        return 0


class DetectionNode(Node):
    def __init__(self, model:IE):
        self.model = model
        input_size = np.sqrt(self.model.get_input_size()/3)
        det_preprocess[0]['resize']['size'] = [input_size, input_size]
        self.det_preprocess = PreProcessingCompose(det_preprocess)
        self.det_postprocess = DetPostProcess(
            thresh=0.3,
            box_thresh=0.6,
            max_candidates=1000,
            unclip_ratio=1.5,
            use_dilation=False,
            score_mode="fast",
            box_type="quad",
        )
    
    def __call__(self, img):
        engine_latency = 0
        input = self.det_preprocess(img)
        input = self.prepare_input(input)
        start_time = time.time()
        req_output = self.model.run_async([input])
        output = self.model.wait(req_output)
        engine_latency += time.time() - start_time
        det_output = self.postprocess(output[0], img.shape)
        return det_output, engine_latency
    
    def postprocess(self, outputs, image_shape):
        # image shape : image h, image w, image c
        dt_boxes = self.det_postprocess(outputs, image_shape)
        dt_boxes = dt_boxes[0]["points"]
        dt_boxes = filter_tag_det_res(dt_boxes, image_shape)
        return dt_boxes

class ClassificationNode(Node):
    def __init__(self, model:IE):
        self.model = model
        self.cls_preprocess = PreProcessingCompose(cls_preprocess)
        self.cls_postprocess = ClsPostProcess()
    
    def __call__(self, det_outputs:List[np.ndarray]):
        # input = self.cls_preprocess(x)
        # input = self.prepare_input(input)
        outputs = []
        engine_latency = 0
        for det_output in det_outputs:
            cls_input = self.prepare_input(det_output)
            start_time = time.time()
            req_output = self.model.run_async([cls_input])
            output = self.model.wait(req_output)
            engine_latency += time.time() - start_time
            label, score = self.cls_postprocess(output[0])[0]
            outputs.append([label, score])
        return outputs, engine_latency

class RecognitionNode(Node):
    def __init__(self, models:dict, version:str):
        self.rec_model_map:dict = models
        
        self.ratio_interval = 10
        self.max_ratio = 30
        self.drop = 0.3

        self.rec_preprocess_map = {}
        for i in range(self.ratio_interval // 2, self.max_ratio, self.ratio_interval):
            ratio_rec_preprocess = rec_preprocess.copy()
            ratio_rec_preprocess[0]['resize']['size'] = [48, 48 * i]
            self.rec_preprocess_map[i] = PreProcessingCompose(ratio_rec_preprocess)
        
        txt_file_name = 'model_files/ppocr_keys_v1.txt'
        if version == 'v5':
            txt_file_name = 'model_files/ppocrv5_dict.txt'
        character_dict_path = os.path.join(engine_path, txt_file_name)
        
        self.rec_postprocess = RecLabelDecode(character_dict_path=character_dict_path, use_space_char=True)
        self.router = self.rec_router
    
    def split_bbox_for_recognition(self, bbox, rec_image_shape, overlap_ratio=0.1):
        return split_bbox_for_recognition(bbox, rec_image_shape, overlap_ratio)
    
    def rec_router(self, width, height, ratio_interval=10, ratio_max=30):
        return rec_router(width, height, ratio_interval, ratio_max)
    
    def merge_recognition_results(self, split_results, overlap_ratio=0.1):
        return merge_recognition_results(split_results, overlap_ratio)
    
    def rotate_if_vertical(self, crop):
        h, w = crop.shape[:2]
        if h > w * 2:
            return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return crop

    def __call__(self, original_image, boxes:List, crops:List[np.ndarray]):
        # input = self.cls_preprocess(x)
        # input = self.prepare_input(input)
        outputs = []
        engine_latency = 0
        min_latency = 10000000
        for i in range(len(crops)):
            h, w = crops[i].shape[:2] # 51, 443
            bbox = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            
            split_info = []
            img_crop_list = []
            split_bboxes = self.split_bbox_for_recognition(bbox, [3, 48, 48 * 40])
            split_info.append(len(split_bboxes))
            for split_bbox in split_bboxes:
                split_bbox = np.array(split_bbox, dtype=np.float32)
                x1, y1 = split_bbox[0]
                x2, y2 = split_bbox[2]
                split_img = crops[i][int(y1):int(y2), int(x1):int(x2)]
                split_img = self.rotate_if_vertical(split_img)
                img_crop_list.append(split_img)

            rec_res = []
            for idx in range(len(img_crop_list)):
                cropped_img = img_crop_list[idx]
                mapped_ratio, mapped_height = self.router(cropped_img.shape[1], cropped_img.shape[0], self.ratio_interval, self.max_ratio)
                inp = self.rec_preprocess_map[mapped_ratio](cropped_img)
                rec_input = self.prepare_input(inp)
                start_time = time.time()
                req_output = self.rec_model_map[mapped_ratio][mapped_height].run_async([rec_input])
                output = self.rec_model_map[mapped_ratio][mapped_height].wait(req_output)
                end_time = time.time()
                if min_latency > end_time - start_time:
                    min_latency = end_time - start_time
                res = self.rec_postprocess(output[0])[0]
                engine_latency += end_time - start_time
                if res[1] > self.drop:
                    rec_res.append(res)

            merged_text, merged_score = self.merge_recognition_results(rec_res)
            merged_res = [(merged_text, merged_score)]

            outputs.append(merged_res)
        return outputs, engine_latency, min_latency

        
class PaddleOcr():
    def __init__(self, det_model:IE, cls_model:IE, rec_models:dict, version:str):
        '''
        @brief:
        @param:
            det_model_path: str, the path of the detection model : det.dxnn
            cls_model_path: str, the path of the classification model : cls.dxnn
            rec_model_path: str, the path of the recognition model
                                    rec_ratio_5_height_10.dxnn
                                    rec_ratio_25_height_30.dxnn
                                    rec_ratio_25_height_20.dxnn
                                    rec_ratio_25_height_10.dxnn
                                    rec_ratio_5_height_30.dxnn
                                    rec_ratio_15_height_30.dxnn
                                    rec_ratio_15_height_10.dxnn
                                    rec_ratio_5_height_20.dxnn
                                    rec_ratio_15_height_20.dxnn
        @return:
            None
        '''
        self.det_model = det_model
        self.cls_model = cls_model
        self.rec_models:dict = rec_models
        
        self.detection_node = DetectionNode(self.det_model)
        self.classification_node = ClassificationNode(self.cls_model)
        self.recognition_node = RecognitionNode(self.rec_models, version)
        self.cls_thresh = 0.9

        self.detection_time_duration = 0
        self.classification_time_duration = 0
        self.recognition_time_duration = 0
        self.min_recognition_time_duration = 0

        self.ocr_run_count = 0

    
    def __call__(self, img):
        det_outputs, engine_latency = self.detection_node(img)
        boxes = self.sorted_boxes(det_outputs)
        self.detection_time_duration += engine_latency * 1000

        # Convert boxes to (n, 4, 2) format if needed
        boxes = self.convert_boxes_to_quad_format(boxes)
        crops = [self.get_rotate_crop_image(img, box) for box in boxes]
        boxes = [box.tolist() for box in boxes]
        cls_results, engine_latency = self.classification_node(crops) # crop 된 이미지를 회전 하는지 유추하는 classification model
        self.classification_time_duration += engine_latency / len(crops) * 1000
        for i, [label, score] in enumerate(cls_results):
            if "180" in label and score > self.cls_thresh:
                crops[i] = cv2.rotate(crops[i], cv2.ROTATE_180)
        rec_results, engine_latency, min_latency = self.recognition_node(img, boxes, crops)
        self.recognition_time_duration += engine_latency / len(crops) * 1000
        self.min_recognition_time_duration += min_latency * 1000
        self.ocr_run_count += 1
        return boxes, crops, rec_results
    
    @staticmethod
    def convert_boxes_to_quad_format(boxes):
        """
        Convert boxes from (n*4, 2) format to (n, 4, 2) format
        @param boxes: numpy array of shape (n*4, 2) or list of boxes where n is number of text boxes
        @return: numpy array of shape (n, 4, 2) where each box has 4 corner points
        """
        # Convert list to numpy array if needed
        if isinstance(boxes, list):
            if len(boxes) == 0:
                return np.array([])
            # Check if each element is already a box (4 points)
            if len(boxes[0]) == 4:
                # Already in correct format, just convert to numpy
                return np.array(boxes)
            else:
                # Flatten the list and convert to numpy
                boxes = np.array(boxes)
        
        if len(boxes.shape) == 2 and boxes.shape[1] == 2:
            # Check if the number of points is divisible by 4
            if boxes.shape[0] % 4 != 0:
                raise ValueError(f"Number of points ({boxes.shape[0]}) must be divisible by 4")
            
            # Reshape to (n, 4, 2) where n = boxes.shape[0] // 4
            num_boxes = boxes.shape[0] // 4
            boxes_reshaped = boxes.reshape(num_boxes, 4, 2)
            return boxes_reshaped
        elif len(boxes.shape) == 3 and boxes.shape[1] == 4 and boxes.shape[2] == 2:
            # Already in correct format
            return boxes
        else:
            raise ValueError(f"Unexpected box format: {boxes.shape}")
    
    @staticmethod
    def sorted_boxes(dt_boxes: np.ndarray) -> List[np.ndarray]:
        boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(boxes)
        for i in range(len(_boxes)-1):
            for j in range(i, -1, -1):
                    if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                        _boxes[j + 1][0][0] < _boxes[j][0][0]
                    ):
                        tmp = _boxes[j]
                        _boxes[j] = _boxes[j + 1]
                        _boxes[j + 1] = tmp
                    else:
                        break
        return _boxes
    
    @staticmethod
    def get_rotate_crop_image(img, points):
        return get_rotate_crop_image(img, points)
