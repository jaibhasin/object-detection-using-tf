import cv2
import torch
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import os
# MoviePy imports (compatible with 2.x)
try:
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
except ImportError:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import time

# ------------------------
# MODEL PATHS
# ------------------------
MODEL_PATH = "models/research/object_detection/yolov8s_oiv7/yolov8s-oiv7.pt"
VIDEO_PATH = "videoplayback.mp4"
OUTPUT_PATH = "voice123_temp.mp4"
OUTPUT_WITH_AUDIO = "voice883_final.mp4"

# Process depth estimation every N frames for speed
DEPTH_FRAME_SKIP = 5

# ------------------------
# NAVIGATION ZONES
# ------------------------
CENTER_ZONE_START = 0.30
CENTER_ZONE_END = 0.70

# ------------------------
# PRIORITY OBJECTS
# ------------------------
PRIORITY_HIGH = {
    'Person', 'Man', 'Woman', 'Girl', 'Boy',
    'Car', 'Bus', 'Truck', 'Motorcycle', 'Van', 'Bicycle',
    'Stairs', 'Door',
}

PRIORITY_MEDIUM = {
    'Traffic light', 'Traffic sign', 'Stop sign',
    'Fire hydrant', 'Parking meter',
    'Chair', 'Table', 'Bench',
}

PRIORITY_LOW = {
    'Building', 'House', 'Tree', 'Skyscraper', 'Window',
    'Handbag', 'Backpack', 'Mobile phone', 'Bottle',
}

ALLOWED = PRIORITY_HIGH | PRIORITY_MEDIUM | PRIORITY_LOW

# ------------------------
# VOICE ALERT SETTINGS
# ------------------------
ALERT_COOLDOWN = 3.0  # Seconds between alerts for same object type
MIN_ALERT_DISTANCE = 150

# ------------------------
# LOAD MODELS
# ------------------------
print("🔄 Loading YOLO model...")
yolo = YOLO(MODEL_PATH)

print("🔄 Loading MiDaS depth estimation model...")
device = torch.device("cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

print("🔄 Initializing audio system...")
# Don't initialize pygame mixer - we don't need it for recording
# pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

print("✅ All models loaded successfully!")

# ------------------------
# VOICE ALERT SYSTEM (with Audio Recording)
# ------------------------
class VoiceAlert:
    def __init__(self):
        self.last_alert_time = {}
        self.speaking = False
        self.audio_clips = []  # Store (timestamp, audio_file) tuples
        self.fps = 30
        self.temp_dir = tempfile.mkdtemp()
        
    def should_alert(self, object_type, depth_value):
        """Check if we should alert about this object"""
        if depth_value < MIN_ALERT_DISTANCE:
            return False
            
        current_time = time.time()
        key = f"{object_type}_{int(depth_value/20)}"
        
        if key in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[key]
            if time_since_last < ALERT_COOLDOWN:
                return False
        
        self.last_alert_time[key] = current_time
        return True
    
    def speak_and_record(self, message, frame_number):
        """Generate speech using gTTS (don't play it during processing)"""
        if self.speaking:
            return
            
        self.speaking = True
        try:
            # Generate speech using Google TTS
            audio_file = os.path.join(self.temp_dir, f"alert_{frame_number}.mp3")
            tts = gTTS(text=message, lang='en', slow=False)
            tts.save(audio_file)
            
            # Calculate timestamp in seconds
            timestamp = frame_number / self.fps
            self.audio_clips.append((timestamp, audio_file))
            
            print(f"🔊 ALERT at {timestamp:.2f}s: {message}")
            
            # DON'T play audio during processing - just record it
            # This speeds up video processing significantly
            
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.speaking = False
    
    def merge_audio_with_video(self, video_path, output_path):
        """Combine all audio alerts with the video using ffmpeg directly"""
        print("\n🎵 Merging audio alerts with video...")
        
        try:
            if not self.audio_clips:
                print("⚠️  No audio alerts to merge. Copying video as-is.")
                import shutil
                shutil.copy(video_path, output_path)
                return
            
            # Create a single audio track by combining all alerts with silence
            print(f"📝 Processing {len(self.audio_clips)} audio alerts...")
            
            from pydub import AudioSegment
            
            # Get video duration
            video = VideoFileClip(video_path)
            video_duration_ms = int(video.duration * 1000)
            video.close()
            
            # Create silent audio track for entire video
            silent_track = AudioSegment.silent(duration=video_duration_ms)
            
            # Overlay each alert at its timestamp
            for timestamp, audio_file in self.audio_clips:
                try:
                    alert_audio = AudioSegment.from_mp3(audio_file)
                    position_ms = int(timestamp * 1000)
                    
                    # Overlay the alert at the correct position
                    silent_track = silent_track.overlay(alert_audio, position=position_ms)
                    print(f"  ✓ Added alert at {timestamp:.2f}s")
                except Exception as e:
                    print(f"  ✗ Failed to add alert at {timestamp:.2f}s: {e}")
            
            # Save combined audio
            combined_audio_path = os.path.join(self.temp_dir, "combined_audio.mp3")
            silent_track.export(combined_audio_path, format="mp3")
            print(f"✅ Combined audio track created")
            
            # Use ffmpeg to merge video with audio
            print("🎬 Merging video with audio track...")
            import subprocess
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', combined_audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                output_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Video with audio saved: {output_path}")
            else:
                print(f"⚠️  FFmpeg error: {result.stderr}")
                print("⚠️  Saving video without audio...")
                import shutil
                shutil.copy(video_path, output_path)
            
            # Cleanup
            try:
                os.remove(combined_audio_path)
            except:
                pass
                
        except ImportError:
            print("❌ pydub not installed. Installing now...")
            import subprocess
            subprocess.run(['pip', 'install', 'pydub'])
            print("✅ Please run the script again")
        except Exception as e:
            print(f"❌ Error merging audio: {e}")
            print("💡 Copying video without audio...")
            import shutil
            shutil.copy(video_path, output_path)
        finally:
            # Cleanup temp files
            for _, audio_file in self.audio_clips:
                try:
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                except:
                    pass

voice_alert = VoiceAlert()

# ------------------------
# DEPTH ESTIMATION
# ------------------------
def estimate_depth(frame):
    """Estimate depth map from frame using MiDaS"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return depth_map

def get_box_depth(depth_map, x1, y1, x2, y2):
    """Calculate average depth in bounding box"""
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    return np.mean(roi)

def get_distance_label(depth_value):
    """Convert depth to distance label"""
    if depth_value > 170:
        return "VERY CLOSE", (0, 0, 255)
    elif depth_value > 128:
        return "CLOSE", (0, 165, 255)
    elif depth_value > 85:
        return "MEDIUM", (0, 255, 255)
    else:
        return "FAR", (0, 255, 0)

# ------------------------
# NAVIGATION LOGIC
# ------------------------
def get_object_zone(x1, x2, frame_width):
    """Determine which zone the object is in"""
    center_x = (x1 + x2) / 2
    relative_pos = center_x / frame_width
    
    if relative_pos < CENTER_ZONE_START:
        return "LEFT"
    elif relative_pos > CENTER_ZONE_END:
        return "RIGHT"
    else:
        return "CENTER"

def get_priority(label_name):
    """Get object priority level"""
    if label_name in PRIORITY_HIGH:
        return "HIGH"
    elif label_name in PRIORITY_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"

def generate_voice_alert(label_name, distance_label, zone):
    """Generate natural voice alert message"""
    if zone == "CENTER":
        if distance_label == "VERY CLOSE":
            return f"Warning! {label_name} directly ahead!"
        elif distance_label == "CLOSE":
            return f"{label_name} ahead"
    
    return None

# ------------------------
# MAIN VIDEO PIPELINE
# ------------------------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    voice_alert.fps = fps  # Set FPS for timestamp calculation
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    print("🎥 Starting Smart Navigation Assistant...")
    print(f"⚡ Processing depth every {DEPTH_FRAME_SKIP} frames")
    print(f"🔊 Voice alerts enabled with audio recording")
    
    frame_count = 0
    depth_map = None

    def draw_zones(frame):
        h, w = frame.shape[:2]
        left_line = int(w * CENTER_ZONE_START)
        right_line = int(w * CENTER_ZONE_END)
        
        cv2.line(frame, (left_line, 0), (left_line, h), (255, 255, 0), 2)
        cv2.line(frame, (right_line, 0), (right_line, h), (255, 255, 0), 2)
        
        cv2.putText(frame, "LEFT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "CENTER PATH", (left_line + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "RIGHT", (right_line + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        if frame_count % DEPTH_FRAME_SKIP == 1 or depth_map is None:
            depth_map = estimate_depth(frame)
        
        results = yolo(frame, verbose=False)[0]
        annotated = frame.copy()
        
        draw_zones(annotated)
        
        center_zone_objects = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label_name = yolo.names[cls_id]

            if label_name not in ALLOWED:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            avg_depth = get_box_depth(depth_map, x1, y1, x2, y2)
            distance_label, color = get_distance_label(avg_depth)
            zone = get_object_zone(x1, x2, width)
            priority = get_priority(label_name)
            
            if zone == "CENTER" and avg_depth > MIN_ALERT_DISTANCE:
                center_zone_objects.append({
                    'label': label_name,
                    'depth': avg_depth,
                    'distance': distance_label,
                    'priority': priority,
                    'zone': zone
                })
            
            label = f"{label_name} {conf:.2f} | {distance_label}"
            if zone == "CENTER":
                label += " [PATH]"
            
            thickness = 3 if zone == "CENTER" else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if center_zone_objects:
            priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            center_zone_objects.sort(
                key=lambda x: (priority_order[x['priority']], x['depth']), 
                reverse=True
            )
            
            for obj in center_zone_objects[:1]:
                # Check distance threshold here before alerting
                if obj['depth'] < MIN_ALERT_DISTANCE:
                    continue
                    
                alert_msg = generate_voice_alert(obj['label'], obj['distance'], obj['zone'])
                if alert_msg and voice_alert.should_alert(obj['label'], frame_count):
                    voice_alert.speak_and_record(alert_msg, frame_count)

        info_text = f"Frame: {frame_count}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(annotated)
        cv2.imshow("Smart Navigation Assistant", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Initial video saved: {OUTPUT_PATH}")
    print(f"📊 Total frames processed: {frame_count}")
    print(f"🎤 Total alerts generated: {len(voice_alert.audio_clips)}")
    
    # Merge audio with video
    voice_alert.merge_audio_with_video(OUTPUT_PATH, OUTPUT_WITH_AUDIO)
    
    # Cleanup temp video
    try:
        os.remove(OUTPUT_PATH)
        print(f"🗑️  Cleaned up temporary video file")
    except:
        pass


if __name__ == "__main__":
    main()