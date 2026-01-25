import cv2
import mediapipe as mp
import numpy as np
import time
from config import POSE_CONNECTIONS, MODEL_PATH


class PoseDetector:
    def __init__(self,camera_idx = 0, running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                 num_poses=1, min_pose_detection_confidence=0.5,
                 min_pose_presence_confidence=0.5, min_tracking_confidence=0.5):
        
        # Global 
        self.running_mode = running_mode
        self.latest_result = None
        self.cap =cv2.VideoCapture(camera_idx)

        # Import MediaPipe Tasks
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker


        def result_callback(result, output_image, timestamp_ms):
            self.latest_result = result

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=result_callback
        )

        self.pose_landmarker = PoseLandmarker.create_from_options(options)
    
    def draw(self, image, landmarks):
        for connection in POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            if self.latest_result and self.latest_result.pose_landmarks:
                landmarks = self.latest_result.pose_landmarks[0]
                h, w, c = image.shape
                
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    
                    if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                        x1, y1 = int(start_point.x * w), int(start_point.y * h)
                        x2, y2 = int(end_point.x * w), int(end_point.y * h)
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image
    
    def detect(self, image, timestamp_ms):
        #Processing image
        try:
            imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format= mp.ImageFormat.SRGB, data= imageRGB)
            
            #Detect landmarks
            if self.running_mode is not mp.tasks.vision.RunningMode.LIVE_STREAM:
                raise TypeError("Only support livestream type")
            

            return self.pose_landmarker.detect_async(mp_image,timestamp_ms)

        except TypeError as e:
            print(f"ERROR: {e}")
            SystemExit

    def process(self, pose_landmarks, image, draw = True):
        landmarks_list = []
        for landmark in pose_landmarks: 
            if draw:
                self.draw(image, landmark)
            landmarks_list.append(landmark)
        return landmarks_list, image
    


if __name__ == "__main__":
    detector = PoseDetector()
    print("VinaFit Pose Detector Initialized")

    while detector.cap.isOpened():
        success, img = detector.cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # 2. Send to MediaPipe (Async) 
        #using normal BGR image from cv2 here
        timestamp_ms = int(time.time() * 1000)
        detector.detect(img, timestamp_ms)

        # 3. Get and draw results
        if detector.latest_result and detector.latest_result.pose_landmarks:
            
            landmarks, annotated_image = detector.process(
                detector.latest_result.pose_landmarks,
                img,
                draw=True
            )
            cv2.imshow("VinaFit Pose Detection", annotated_image)
    
        else:
            cv2.imshow("VinaFit Pose Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

