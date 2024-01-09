
import cv2 
import os 
import time 
import uuid 
import multiprocessing
import numpy as np 
import onnxruntime as ort 

from multiprocessing import Queue, Process
from typing import Any

def init_session(onnx_path,use_gpu):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess

class PickableInferenceSession: # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, onnx_path, use_gpu):
        self.onnx_path = onnx_path 
        self.use_gpu = use_gpu
        self.sess = init_session(self.onnx_path, use_gpu)

    def run(self, *args):
        return self.sess.run(*args)

    def get_inputs(self):
        return self.sess.get_inputs()

    def get_outputs(self):
        return self.sess.get_outputs()

    def __getstate__(self):
        return {'onnx_path': self.onnx_path, 'use_gpu': self.use_gpu}

    def __setstate__(self, values):
        self.onnx_path = values['onnx_path']
        self.use_gpu = values['use_gpu']
        self.sess = init_session(self.onnx_path, self.use_gpu)

class TruckDetection:
    def __init__(self, 
                 onnx_path,
                 scale = 1.0 / 255.0,
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225],
                 threshold = 0.5,
                 labels = ['no_car','contain_car'],
                 use_gpu = True):

        # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        # opts = ort.SessionOptions()
        # opts.enable_profiling = False
        # opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL 
        # self.ort_session = ort.InferenceSession(onnx_path,opts,providers=providers)
        # print(self.ort_session.get_providers())

        self.ort_session = PickableInferenceSession(onnx_path,use_gpu)
        
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        self.scale = scale
        self.mean = np.array(mean).reshape(1,1,3).astype('float32')
        self.std = np.array(std).reshape(1,1,3).astype('float32')

        self.threshold = threshold
        self.labels = labels

    def _preprocess_data(self, img):
        # 数据预处理逻辑
        # rgb 
        img = img[:,:,::-1]
        # reisze 224 
        img = cv2.resize(img, (224, 224),interpolation=cv2.INTER_LINEAR)
        # normalize 
        img = (img.astype('float32') * self.scale - self.mean) / self.std
        # to CHW
        img = img.transpose((2,0,1))
        # batch 
        img = np.expand_dims(img, axis=0)
        return img

    def _postprocess_pred(self, probs):
        # 预测结果的后处理逻辑
        score = probs.squeeze()[1]
        result = None
        if score < self.threshold:
            result = {
                "class_ids": 0,
                "scores": 1 - score,
                "label_name": self.labels[0]
            }
        else:
            result = {
                "class_ids": 1,
                "scores": score,
                "label_name": self.labels[1]
            }
        return result 

    def __call__(self, img):
        # 将数据转换为模型输入所需的格式
        input_data = self._preprocess_data(img)
        
        # 使用ONNX运行时进行预测
        outputs = self.ort_session.run([self.output_name], {self.input_name: input_data})   
        # 处理预测结果
        predictions = self._postprocess_pred(outputs[0])
        return predictions

class PlateRecognition:
    def __init__(self, url:str):
        self.url = url 
    
    def __call__(self, img) -> Any:
       pass  

class TruckState:
    INITIAL = 0  # 初始
    DOCKED = 1  # 停靠
    DEPARTED = 2  # 离港
    SAILING = 3 # 行驶中

class TruckInfo:
    def __init__(self):
        self.enter_time = None
        self.leave_time = None
        self.truck_state = TruckState.EMPTY

        self.seq_plate_info = []  # list[dict]
        self.last_plate_info = None  
        self.plate_reg_times = 0 
        self.max_plate_reg_times = 10

    def reset(self):
        self.enter_time = None
        self.leave_time = None
        self.cur_state = TruckState.INITIAL
        self.plate_info = []
        self.plate_reg_times = 0

    def update_truck_state(self, cur_time:str, car_exist:bool, car_no:str):
        pass 

class TruckMonitor(Process):
    def __init__(self,
                 q_car_exist 
                 ):
        super(Process, self).__init__()

        self.q_car_exist = q_car_exist
        self.detect_truck = TruckDetection(r"d:\Code\PaddleClas\output\PPLCNet_x1_0\inference20240108.onnx",
                                           threshold = 0.5, use_gpu=True)
        self.running = True

    def _detect(self, img):
        res = self.detect_truck(img)
        return res 

    def run(self):
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 800, 600)

        x, y, w, h, use_roi = 0, 0, 0, 0, False
        count = 0 
        res, exist_car_info = {'class_ids': -1, 'scores': -1, 'label_name': 'None'} , 'None'
        while self.running:
            frame = self.q_car_exist.get()
            count += 1
            draw_frame = np.copy(frame)

            if use_roi:
                if count % 25 == 0:
                    beg_time = time.time()
                    crop_img = frame[y:y+h,x:x+w]
                    res = self._detect(crop_img)
                    end_time = time.time()
                    print(res, ' cost_time: {:.4f} ms'.format((end_time - beg_time) * 1000))

                if res['class_ids'] == 1:
                    exist_car_info = 'contain_car score: {:.4}'.format(res['scores'])
                elif res['class_ids'] == 0:
                    exist_car_info = 'no_car socre: {:.4}'.format(res['scores'])
                else:
                    exist_car_info = 'None'

                # draw info
                cv2.rectangle(draw_frame, pt1=(x,y), pt2=(x+w,y+h), color=(0,0,255), thickness=3)

                (font_w, font_h), _ = cv2.getTextSize(exist_car_info, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                draw_frame = cv2.rectangle(draw_frame, pt1=(x, y+h), pt2=(x + font_w, y+h+font_h),
                                        color=(0,0,255) if res['class_ids'] == 1 else (255,0,0), thickness=-1)
                draw_frame = cv2.putText(draw_frame, exist_car_info, (x, y+h+font_h),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)

            cv2.imshow('frame',draw_frame)
            presskey = cv2.waitKey(1) & 0XFF 
            if presskey == ord('q'):
                break 
            elif presskey == ord('a'):
                x, y, w, h = cv2.selectROI(windowName='frame', img=frame)
                use_roi = True
            else:
                pass 


# 24.3760b342fed9dbb91e58bc2c104ff19f.2592000.1707363605.282335-25562841
class PreView:
    def __init__(self, max_q_size = 1000):
        self.q_car_exist = Queue(maxsize = max_q_size)
        self.truckMonitor = TruckMonitor(q_car_exist=self.q_car_exist)

    def __call__(self, video_path):
        self.truckMonitor.start()
        vid = cv2.VideoCapture(video_path)
        fps, width, height = vid.get(5), vid.get(3), vid.get(4)
        print('fps： {} w: {} h: {}'.format(fps, width, height))

        ret, frame = vid.read()
        count = 0 
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                continue 

            if self.q_car_exist.full():
                self.q_car_exist.get()
            self.q_car_exist.put(frame)

        vid.release()
        self.truckMonitor.join()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    review = PreView()
    review(r"rtsp://admin:tenglong12345@117.78.18.244:5541/Streaming/Channels/901")
    # review(r"d:\Code\test\videos\cainiao20240107-0739-1.mp4")
    