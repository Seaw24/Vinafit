from pose_dectection import PoseDetector
import cv2
import time
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFont, ImageDraw  # <--- NEW LIBRARY FOR VIETNAMESE
from config import SQUAT_RESET_THRESHOLD_DESCENDING, SQUAT_RESET_THRESHOLD_ASCENDING, calculate_3_point_angle, calculate_vertical_angle

# --- COLORS (RGB for Pillow) ---
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (200, 200, 200)

class Squat:
    def __init__(self, squat_reset_threshold_descending=SQUAT_RESET_THRESHOLD_DESCENDING, 
                 squat_reset_threshold_ascending=SQUAT_RESET_THRESHOLD_ASCENDING, 
                 camera_idx=0, running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                 num_poses=1, min_pose_detection_confidence=0.7,
                 min_pose_presence_confidence=0.6, min_tracking_confidence=0.6):
        
        self.detector = PoseDetector(camera_idx, running_mode, num_poses,
                                     min_pose_detection_confidence, min_pose_presence_confidence, min_tracking_confidence)
        self.current_state = "Deactivated"
        self.squat_count = 0
        self.feedback = {}  
        self.set_correct = [] 
        self.reset_thresh_descending = squat_reset_threshold_descending
        self.reset_thresh_ascending = squat_reset_threshold_ascending
        self.correct = True
        
        # SWITCH VARIABLES
        self.switch_start_time = None 

    def checking(self, landmarks):
        ankle, knee, hip, shoulder, toe, heel, hand = self.right_or_left(landmarks)
        
        # --- CALCULATE ANGLES ---
        geo_knee = calculate_3_point_angle(hip, knee, ankle) 
        knee_flexion = 180 - geo_knee
        
        geo_hip = calculate_3_point_angle(knee, hip, shoulder)
        hip_flexion = 180 - geo_hip

        # Reset feedback if state changes
        if self.current_state != "Standing":
             self.feedback = {} 

        self.correct = True
        
        # --- STANDING (RESET PHASE) ---
        if knee_flexion < self.reset_thresh_descending and self.current_state in ["Ascending", "Descending", "Bottom"]:
                
                # Check for "Fake Shallow" reps
                if self.current_state == "Descending":
                    self.correct = False
                    self.feedback["Depth"] = "Quá nông! Xuống sâu hơn nhé."
                else:
                    self.feedback["Depth"] = "Tốt! Chân thẳng, chuẩn bị xuống."

                self.squat_count += 1
                self.set_correct.append({"correct": self.correct, "feedback": self.feedback.copy()})

                self.current_state = "Standing" 

        # --- START DESCENDING ---
        elif self.current_state == "Standing" and knee_flexion > self.reset_thresh_descending:
            self.current_state = "Descending"
            self.feedback["Depth"] = "Hít sâu... Xuống chậm thôi!"

        # --- ACTIVE ANALYSIS ---
        if self.current_state == "Descending":
            self.checking_trunk_spine(hip, shoulder)
            self.checking_heel_lift(toe, heel)
            self.checking_depth_and_hip(knee_flexion, hip_flexion)
        elif self.current_state == "Ascending" or self.current_state == "Bottom":
            self.checking_trunk_spine(hip, shoulder)
            self.checking_heel_lift(toe, heel)
        
        # STATE TRANSITION
        if self.current_state == "Bottom" and knee_flexion < 70: 
             self.current_state = "Ascending"

        return self.correct, knee_flexion, hip_flexion

    # --- CHECK FUNCTIONS ---
    def checking_trunk_spine(self, hip, shoulder):
        trunk_angle = calculate_vertical_angle(hip, shoulder)
        if trunk_angle > 28:     
            self.correct = False
            self.feedback["Back"] = "Ưỡn ngực lên! (Lưng quá cong)"

    def checking_heel_lift(self, toe, heel):
        if (toe.y - heel.y) > 0.025: 
                self.correct = False
                self.feedback["Heel"] = "Dồn lực vào gót chân!"

    def checking_depth_and_hip(self,knee_flexion, hip_flexion):
        if knee_flexion < 60:
            self.feedback["Depth"] = "Xuống thấp nữa!"
        elif 60 <= knee_flexion < 80:
            self.feedback["Depth"] = "Gần được rồi... Thêm chút nữa!"
        elif knee_flexion >= 80:
            self.current_state = "Bottom"
            if hip_flexion < 90:
                self.feedback["Hip"] = "Đẩy mông ra sau!" 
                self.correct = False
            elif hip_flexion > 110:
                self.feedback["Hip"] = "Ưỡn ngực lên! (Đừng gập bụng)"
                self.correct = False
            else:
                self.feedback["Depth"] = "TUYỆT VỜI! GIỮ THẾ!"
                self.correct = True

    def right_or_left(self, landmarks):
        if landmarks[26].visibility > landmarks[25].visibility:
            return landmarks[28], landmarks[26], landmarks[24], landmarks[12], landmarks[32], landmarks[30], landmarks[16]
        else:
            return landmarks[27], landmarks[25], landmarks[23], landmarks[11], landmarks[31], landmarks[29], landmarks[15]
    
    # --- ROBUST SWITCH FUNCTION ---
    def switch(self, landmarks):
        # 1. AUTO-STOP AT 15 REPS
        if self.squat_count >= 15 and self.current_state != "Deactivated":
            self.current_state = "Deactivated"
            self.switch_start_time = None
            return "HOÀN THÀNH TẬP LUYỆN!"

        # 2. SAFETY LOCK (Only checks when Deactivated)
        if self.current_state == "Deactivated":
            l_knee_geo = calculate_3_point_angle(landmarks[23], landmarks[25], landmarks[27])
            r_knee_geo = calculate_3_point_angle(landmarks[24], landmarks[26], landmarks[28])
            
            # If knees are bent, LOCK START.
            if (180 - l_knee_geo) > 5 or (180 - r_knee_geo) > 5:
                self.switch_start_time = None
                return "ĐỨNG THẲNG ĐỂ BẮT ĐẦU" 

        # 3. GESTURE CHECK (Hands High)
        nose = landmarks[0]
        l_wrist = landmarks[15]
        r_wrist = landmarks[16]
        hands_high = (l_wrist.y < nose.y) and (r_wrist.y < nose.y)

        if hands_high and self.current_state == "Deactivated":
            if self.switch_start_time is None:
                self.switch_start_time = time.time()
                return "GIỮ YÊN..."
            
            elif time.time() - self.switch_start_time > 1.5: # 1.5s Hold
                if self.current_state == "Deactivated":
                    # --- RESET EVERYTHING ON START ---
                    self.current_state = "Standing"
                    self.squat_count = 0        
                    self.set_correct = []       
                    self.switch_start_time = None
                    return "BẮT ĐẦU!"
                elif self.squat_count >= 15:
                    self.current_state = "Deactivated"
                    self.switch_start_time = None
                    return "ĐÃ DỪNG LẠI"
            
            remaining = 1.5 - (time.time() - self.switch_start_time)
            return f"GIỮ YÊN: {remaining:.1f}s"
        else:
            self.switch_start_time = None
            if self.current_state == "Deactivated":
                return "GIƠ TAY CAO ĐỂ BẮT ĐẦU" 
            return None

def main():
    squat = Squat()
    detector = squat.detector

    # --- LOAD FONTS FOR VIETNAMESE ---
    try:
        # On Windows, this usually works. On Mac/Linux, use path like "/Library/Fonts/Arial.ttf"
        font_path = "arial.ttf" 
        font_large = ImageFont.truetype(font_path, 40)
        font_medium = ImageFont.truetype(font_path, 30)
        font_small = ImageFont.truetype(font_path, 20)
    except IOError:
        print("WARNING: Arial font not found. Vietnamese may not display correctly.")
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    print("VinaFit Started.")
    
    while detector.cap.isOpened():
        success, frame = detector.cap.read()
        if not success: break
        
        # 1. MIRROR FRAME
        frame = cv2.flip(frame, 1)
        
        timestamp_ms = int(time.time() * 1000)
        detector.detect(frame, timestamp_ms)

        if detector.latest_result and detector.latest_result.pose_landmarks:
            landmarks_list = detector.latest_result.pose_landmarks[0]
            
            # Draw Skeleton (OpenCV is faster for lines)
            detector.process([landmarks_list], frame, draw=False)
            
            # Check Status
            switch_status = squat.switch(landmarks_list)
            
            # Calculate Logic
            k_flex, h_flex = 0, 0
            correct = True
            if squat.current_state != "Deactivated":
                correct, k_flex, h_flex = squat.checking(landmarks_list)

            # --- RENDER TEXT WITH PILLOW (FOR VIETNAMESE) ---
            # Convert OpenCV (BGR) to PIL (RGB)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # A. DRAW HUD BACKGROUND (Using cv2 logic converted to PIL drawing)
            # (We draw text directly on top)
            
            # Draw HUD Box (Simulated with PIL rectangle if needed, or stick to CV2 before conversion)
            # Let's use PIL for everything text-related.
            
            # 1. Reps & State
            draw.text((20, 30), f"Reps: {squat.squat_count}", font=font_large, fill=COLOR_WHITE)
            draw.text((20, 80), f"State: {squat.current_state}", font=font_medium, fill=COLOR_GRAY)
            
            # 2. Angles (Only if active)
            if squat.current_state != "Deactivated":
                draw.text((300, 30), f"Knee: {int(k_flex)}", font=font_small, fill=COLOR_GRAY)
                draw.text((300, 60), f"Hip:  {int(h_flex)}", font=font_small, fill=COLOR_GRAY)

                # 3. FEEDBACK (Vietnamese)
                y_pos = 130
                for key, msg in squat.feedback.items():
                    text_color = COLOR_GREEN if correct else COLOR_RED
                    for line in msg.split('\n'):
                        draw.text((20, y_pos), line, font=font_medium, fill=text_color)
                        y_pos += 35

            # 4. SUMMARY (If Deactivated)
            if squat.current_state == "Deactivated" and len(squat.set_correct) > 0:
                draw.text((20, 150), "TỔNG KẾT BUỔI TẬP:", font=font_medium, fill=COLOR_WHITE)
                y_summary = 190
                for i, record in enumerate(squat.set_correct):
                    is_good = record['correct']
                    status = "TỐT" if is_good else "LỖI"
                    c = COLOR_GREEN if is_good else COLOR_RED
                    draw.text((20, y_summary), f"Rep {i+1}: {status}", font=font_small, fill=c)
                    y_summary += 25
                    
                    if not is_good:
                         for k, v in record['feedback'].items():
                             # Filter out "Good" messages from error log
                             if "Tốt" not in v and "Oke" not in v and "Hít sâu" not in v:
                                draw.text((40, y_summary), f"- {v}", font=font_small, fill=COLOR_GRAY)
                                y_summary += 20

            # 5. GESTURE STATUS (Always on top)
            if switch_status:
                if "BẮT ĐẦU" in switch_status or "HOÀN THÀNH" in switch_status: c = COLOR_GREEN
                elif "GIỮ" in switch_status: c = COLOR_YELLOW
                elif "ĐỨNG THẲNG" in switch_status: c = COLOR_RED
                else: c = COLOR_WHITE
                
                # Draw centered text (approx)
                draw.text((400, 400), switch_status, font=font_large, fill=c)

            # Convert PIL back to OpenCV (BGR)
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("Squat Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    detector.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()