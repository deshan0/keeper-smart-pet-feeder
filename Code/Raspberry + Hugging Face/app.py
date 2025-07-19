import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime, timedelta
import json
import firebase_admin
from firebase_admin import credentials, db
import schedule
import threading
import time
import pytz

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8s.pt')
print("Model loaded successfully!")

# Firebase configuration
FIREBASE_URL = "FIREBASE_URL"
FIREBASE_CRED_PATH = "firebase-credentials.json"

# 🕐 Timezone Configuration
LOCAL_TIMEZONE = "Asia/Colombo"  # Change this to your timezone

# Timezone helper functions
def get_local_time():
    """Get current time in configured timezone"""
    try:
        utc_now = datetime.utcnow()
        utc_time = pytz.utc.localize(utc_now)
        local_tz = pytz.timezone(LOCAL_TIMEZONE)
        local_time = utc_time.astimezone(local_tz)
        return local_time
    except Exception as e:
        print(f"❌ Error getting local time: {e}")
        return datetime.utcnow()

def format_local_time(dt=None):
    """Format local time as string"""
    if dt is None:
        dt = get_local_time()
    return dt.strftime("%I:%M %p")

def format_local_timestamp(dt=None):
    """Format local timestamp for logging"""
    if dt is None:
        dt = get_local_time()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# 🍖 Global feeding tracker - FIXED: Use timezone-aware datetime
feeding_tracker = {
    'consecutive_detections': 0,
    'detection_history': [],
    'last_fed_time': None,
    'session_start_time': get_local_time()  # 🔧 FIXED: Use timezone-aware time
}

# Initialize Firebase
def init_firebase():
    """Initialize Firebase connection with complete structure"""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred, {
                'databaseURL': FIREBASE_URL
            })
        
        init_firebase_structure()
        print("✅ Firebase initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        return False

def init_firebase_structure():
    """🍖 Initialize complete Firebase database structure"""
    try:
        ref = db.reference('/')
        current_data = ref.get()
        
        if not current_data:
            current_time = format_local_time()
            initial_data = {
                "DogCame": {
                    "Eaten": False,
                    "Number": 0,
                    "Time": current_time
                },
                "Intruder": {
                    "IsHere": False,
                    "Time": current_time
                },
                "Variables": {
                    "IntruderAlert": False
                },
                "FeedCommand": {
                    "FeedNow": False
                }
            }
            ref.set(initial_data)
            print("🔧 Firebase structure initialized")
        else:
            # Check and create missing structures
            if 'FeedCommand' not in current_data:
                db.reference('FeedCommand').set({"FeedNow": False})
                print("🍖 Added FeedCommand structure to Firebase")
                
    except Exception as e:
        print(f"⚠️ Error initializing Firebase structure: {e}")

def check_intruder_alert():
    """Check IntruderAlert status from Firebase"""
    try:
        ref = db.reference('Variables/IntruderAlert')
        status = ref.get()
        return bool(status) if status is not None else False
    except Exception as e:
        print(f"❌ Error checking IntruderAlert: {e}")
        return False

def check_feed_now():
    """🍖 Check FeedNow status from Firebase"""
    try:
        ref = db.reference('FeedCommand/FeedNow')
        status = ref.get()
        return bool(status) if status is not None else False
    except Exception as e:
        print(f"❌ Error checking FeedNow: {e}")
        return False

def update_dog_count_with_feeding(dogs_detected, feed_now_enabled):
    """🍖 Update dog count with feeding logic"""
    global feeding_tracker
    
    try:
        if dogs_detected > 0:
            number_ref = db.reference('DogCame/Number')
            current_count = number_ref.get() or 0
            new_count = current_count + dogs_detected
            
            current_time = format_local_time()
            dog_ref = db.reference('DogCame')
            dog_ref.update({
                'Number': new_count,
                'Time': current_time
            })
            
            print(f"📊 Updated Firebase: DogCame/Number = {new_count} (+{dogs_detected}) at {current_time}")
        
        if feed_now_enabled:
            should_feed = update_feeding_tracker(dogs_detected)
            consecutive_count = feeding_tracker['consecutive_detections']
            print(f"🍖 Feeding Status: FeedNow=True, Consecutive={consecutive_count}/5")
            
            if should_feed:
                trigger_feeding()
        else:
            feeding_tracker['consecutive_detections'] = 0
            feeding_tracker['detection_history'] = []
            
    except Exception as e:
        print(f"❌ Error updating dog count: {e}")

def update_feeding_tracker(dogs_detected):
    """🍖 Update feeding tracker with consecutive dog detections"""
    global feeding_tracker
    
    has_dogs = dogs_detected > 0
    feeding_tracker['detection_history'].append(has_dogs)
    
    if len(feeding_tracker['detection_history']) > 5:
        feeding_tracker['detection_history'] = feeding_tracker['detection_history'][-5:]
    
    if has_dogs:
        feeding_tracker['consecutive_detections'] += 1
    else:
        feeding_tracker['consecutive_detections'] = 0
    
    history_length = len(feeding_tracker['detection_history'])
    if (history_length >= 5 and 
        all(feeding_tracker['detection_history'][-5:]) and 
        feeding_tracker['consecutive_detections'] >= 5):
        return True
    
    return False

def trigger_feeding():
    """🍖 Trigger feeding: Set DogCame/Eaten = True and start 4-hour timer"""
    global feeding_tracker
    
    try:
        current_time_obj = get_local_time()
        current_time_str = format_local_time(current_time_obj)
        
        dog_ref = db.reference('DogCame')
        dog_ref.update({
            'Eaten': True,
            'Time': current_time_str
        })
        
        feeding_tracker['last_fed_time'] = current_time_obj
        feeding_tracker['consecutive_detections'] = 0
        feeding_tracker['detection_history'] = []
        
        reset_time = current_time_obj + timedelta(hours=4)
        reset_time_str = format_local_time(reset_time)
        
        print(f"🍖 FEEDING TRIGGERED! DogCame/Eaten = True at {current_time_str}")
        print(f"🕐 Auto-reset scheduled in 4 hours ({reset_time_str})")
        
    except Exception as e:
        print(f"❌ Error triggering feeding: {e}")

def update_intruder_status(humans_detected):
    """Update intruder status in Firebase"""
    try:
        current_time = format_local_time()
        
        intruder_data = {
            'IsHere': humans_detected > 0,
            'Time': current_time
        }
        
        intruder_ref = db.reference('Intruder')
        intruder_ref.set(intruder_data)
        
        if humans_detected > 0:
            print(f"🚨 INTRUDER ALERT! Updated Firebase: IsHere=True, Time={current_time}")
        else:
            print(f"✅ No intruders - Firebase updated: IsHere=False, Time={current_time}")
            
    except Exception as e:
        print(f"❌ Error updating intruder status: {e}")

def check_feeding_reset():
    """🍖 Check if 4 hours have passed since last feeding and reset if needed"""
    global feeding_tracker
    
    try:
        last_fed = feeding_tracker['last_fed_time']
        
        if last_fed:
            current_time_obj = get_local_time()
            time_diff = current_time_obj - last_fed
            
            if time_diff >= timedelta(hours=4):
                current_time_str = format_local_time(current_time_obj)
                print(f"\n🍖 4 HOURS PASSED - Auto-resetting Eaten status at {current_time_str}...")
                
                dog_ref = db.reference('DogCame/Eaten')
                dog_ref.set(False)
                
                time_ref = db.reference('DogCame/Time')
                time_ref.set(current_time_str)
                
                feeding_tracker['last_fed_time'] = None
                feeding_tracker['consecutive_detections'] = 0
                feeding_tracker['detection_history'] = []
                
                print(f"✅ Auto-reset completed: DogCame/Eaten = False at {current_time_str}")
                return True
    
    except Exception as e:
        print(f"❌ Error in feeding reset check: {e}")
    
    return False

def reset_daily_counters():
    """Reset daily counters at end of day"""
    global feeding_tracker
    
    try:
        current_time = format_local_time()
        print(f"\n🔄 DAILY RESET - Resetting counters at {current_time}...")
        
        dog_ref = db.reference('DogCame')
        dog_ref.update({
            'Eaten': False,
            'Number': 0,
            'Time': current_time
        })
        
        intruder_ref = db.reference('Intruder')
        intruder_ref.update({
            'IsHere': False,
            'Time': current_time
        })
        
        feeding_tracker['consecutive_detections'] = 0
        feeding_tracker['detection_history'] = []
        feeding_tracker['last_fed_time'] = None
        
        print(f"✅ Daily counters reset successfully at {current_time}!")
        
    except Exception as e:
        print(f"❌ Error resetting daily counters: {e}")

def run_scheduler():
    """Background scheduler for resets"""
    while True:
        schedule.run_pending()
        time.sleep(60)

# Schedule operations
schedule.every().day.at("23:59").do(reset_daily_counters)
schedule.every().hour.do(check_feeding_reset)

# Start background scheduler
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()
print("⏰ Scheduler started for daily reset and feeding checks")

# Initialize global storage
detection_history = []
history_file = "detection_history.json"
images_folder = "history_images"

os.makedirs(images_folder, exist_ok=True)
firebase_available = init_firebase()

def load_history():
    """Load detection history from file"""
    global detection_history
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                loaded_data = json.load(f)
                detection_history = loaded_data if isinstance(loaded_data, list) else []
            print(f"📚 Loaded {len(detection_history)} previous detections")
        else:
            detection_history = []
    except Exception as e:
        print(f"⚠️ Could not load history: {e}")
        detection_history = []

def save_history():
    """Save detection history to file"""
    global detection_history
    try:
        with open(history_file, 'w') as f:
            json.dump(detection_history, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save history: {e}")

def save_image_to_history(image_array, timestamp, has_pets=False, has_humans=False):
    """Save processed image to history folder"""
    try:
        if isinstance(image_array, np.ndarray):
            pil_image = Image.fromarray(image_array)
        else:
            pil_image = image_array
        
        safe_timestamp = timestamp.replace(":", "-").replace(" ", "_")
        
        if has_pets and has_humans:
            prefix = "PETS_HUMANS_"
        elif has_pets:
            prefix = "PETS_"
        elif has_humans:
            prefix = "HUMANS_"
        else:
            prefix = "NOTHING_"
        
        filename = f"{prefix}{safe_timestamp}.jpg"
        filepath = os.path.join(images_folder, filename)
        
        pil_image.save(filepath, "JPEG", quality=90)
        return filepath
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return None

def get_history_images():
    """Get list of history images for gallery"""
    try:
        image_files = []
        if os.path.exists(images_folder):
            files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(images_folder, x)), reverse=True)
            image_files = [os.path.join(images_folder, f) for f in files[:20]]
        return image_files
    except Exception as e:
        print(f"❌ Error getting history images: {e}")
        return []

def predict(image):
    """🔥 ENHANCED: Detect pets and humans with complete Firebase integration"""
    global detection_history, feeding_tracker
    
    if image is None:
        return None, "❌ No image provided", get_history_text(), get_history_images()
    
    try:
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        print(f"Processing image of shape: {image_np.shape}")
        
        # Check Firebase statuses
        intruder_alert_enabled = False
        feed_now_enabled = False
        
        if firebase_available:
            intruder_alert_enabled = check_intruder_alert()
            feed_now_enabled = check_feed_now()
        
        # Run YOLO inference
        results = model(image_np)
        
        pet_classes = [15, 16]  # cat, dog
        human_classes = [0]     # person
        
        pet_detections = []
        human_detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in pet_classes and confidence > 0.3:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        class_name = model.names[class_id]
                        
                        pet_detections.append({
                            'class': class_name,
                            'confidence': round(confidence, 2),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                    
                    elif class_id in human_classes and confidence > 0.3 and intruder_alert_enabled:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        class_name = model.names[class_id]
                        
                        human_detections.append({
                            'class': class_name,
                            'confidence': round(confidence, 2),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        # Update Firebase
        if firebase_available:
            dogs_detected = sum(1 for det in pet_detections if det['class'] == 'dog')
            humans_detected = len(human_detections)
            
            if dogs_detected > 0 or feed_now_enabled:
                update_dog_count_with_feeding(dogs_detected, feed_now_enabled)
            
            if intruder_alert_enabled:
                update_intruder_status(humans_detected)
        
        # Get annotated image
        annotated_image = results[0].plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Create result text
        timestamp = format_local_timestamp()
        
        result_text = f"🔍 **SMART DETECTION RESULTS**\n"
        result_text += f"📅 Processed at: {timestamp} ({LOCAL_TIMEZONE})\n"
        result_text += f"🔥 IntruderAlert: {'ENABLED' if intruder_alert_enabled else 'DISABLED'}\n"
        result_text += f"🍖 FeedNow: {'ENABLED' if feed_now_enabled else 'DISABLED'}\n\n"
        
        if pet_detections:
            result_text += f"🐕 **PETS FOUND: {len(pet_detections)}**\n"
            dogs_count = sum(1 for det in pet_detections if det['class'] == 'dog')
            
            for i, det in enumerate(pet_detections, 1):
                result_text += f"  {i}. {det['class'].title()}: {det['confidence']:.2f}\n"
            
            if feed_now_enabled and dogs_count > 0:
                consecutive = feeding_tracker['consecutive_detections']
                result_text += f"\n🍖 **FEEDING STATUS:**\n"
                result_text += f"  Consecutive Detections: {consecutive}/5\n"
                result_text += f"  Detection History: {feeding_tracker['detection_history']}\n"
                if feeding_tracker['last_fed_time']:
                    time_since_fed = get_local_time() - feeding_tracker['last_fed_time']
                    hours_since = time_since_fed.total_seconds() / 3600
                    result_text += f"  Last Fed: {format_local_time(feeding_tracker['last_fed_time'])} ({hours_since:.1f}h ago)\n"
                else:
                    result_text += f"  Last Fed: Never\n"
        else:
            result_text += f"🐕 **PETS: None detected**\n"
        
        result_text += "\n"
        
        if intruder_alert_enabled:
            if human_detections:
                result_text += f"👤 **HUMANS FOUND: {len(human_detections)} ⚠️**\n"
                for i, det in enumerate(human_detections, 1):
                    result_text += f"  {i}. {det['class'].title()}: {det['confidence']:.2f}\n"
                result_text += f"🚨 **INTRUDER STATUS: Updated in Firebase**\n"
            else:
                result_text += f"👤 **HUMANS: None detected ✅**\n"
                result_text += f"✅ **INTRUDER STATUS: Cleared in Firebase**\n"
        else:
            result_text += f"👤 **HUMANS: Detection disabled (IntruderAlert=False)**\n"
        
        # Save to history
        has_pets = len(pet_detections) > 0
        has_humans = len(human_detections) > 0
        image_path = save_image_to_history(annotated_image, timestamp, has_pets, has_humans)
        
        history_entry = {
            "timestamp": timestamp,
            "pet_detections": pet_detections,
            "human_detections": human_detections,
            "total_pets": len(pet_detections),
            "total_humans": len(human_detections),
            "dogs_detected": sum(1 for det in pet_detections if det['class'] == 'dog'),
            "intruder_alert_enabled": intruder_alert_enabled,
            "feed_now_enabled": feed_now_enabled,
            "feeding_consecutive": feeding_tracker['consecutive_detections'],
            "image_path": image_path,
            "has_pets": has_pets,
            "has_humans": has_humans
        }
        
        if detection_history is None:
            detection_history = []
        
        detection_history.append(history_entry)
        
        if len(detection_history) > 50:
            detection_history = detection_history[-50:]
        
        save_history()
        
        return annotated_image, result_text, get_history_text(), get_history_images()
    
    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        print(error_msg)
        return None, error_msg, get_history_text(), get_history_images()

def get_history_text():
    """Get formatted history text with statistics - FIXED timezone issue"""
    global detection_history, feeding_tracker
    
    if detection_history is None:
        detection_history = []
    
    if not detection_history:
        history_text = "📝 No detections yet. Upload an image to start detecting pets and humans!\n\n"
    else:
        history_text = f"📊 **DETECTION HISTORY** (Last {len(detection_history)} sessions)\n\n"
        
        total_pets = sum(entry.get('total_pets', 0) for entry in detection_history)
        total_humans = sum(entry.get('total_humans', 0) for entry in detection_history)
        total_dogs = sum(entry.get('dogs_detected', 0) for entry in detection_history)
        total_sessions = len(detection_history)
        pet_sessions = sum(1 for entry in detection_history if entry.get('total_pets', 0) > 0)
        human_sessions = sum(1 for entry in detection_history if entry.get('total_humans', 0) > 0)
        
        history_text += f"🎯 **OVERALL STATISTICS:**\n"
        history_text += f"• Total Sessions: {total_sessions}\n"
        history_text += f"• Pet Detection Sessions: {pet_sessions}\n"
        history_text += f"• Human Detection Sessions: {human_sessions}\n"
        history_text += f"• Total Pets Found: {total_pets} (Dogs: {total_dogs})\n"
        history_text += f"• Total Humans Found: {total_humans}\n"
        
        if total_sessions > 0:
            history_text += f"• Pet Success Rate: {(pet_sessions/total_sessions*100):.1f}%\n"
        
        history_text += f"\n📋 **RECENT DETECTIONS:**\n"
        
        recent_entries = detection_history[-10:]
        for i, entry in enumerate(reversed(recent_entries), 1):
            timestamp = entry.get('timestamp', 'Unknown time')
            total_pets = entry.get('total_pets', 0)
            total_humans = entry.get('total_humans', 0)
            
            status_line = f"{i}. {timestamp} - "
            
            if total_pets > 0:
                pet_detections = entry.get('pet_detections', [])
                pets_found = ", ".join([f"{det.get('class', 'Unknown').title()}({det.get('confidence', 0):.2f})" 
                                      for det in pet_detections])
                status_line += f"🐕 {pets_found}"
            else:
                status_line += f"🐕 No pets"
            
            if total_humans > 0:
                status_line += f" | 👤 {total_humans} human(s) ⚠️"
            else:
                status_line += f" | 👤 No humans"
            
            history_text += status_line + "\n"
    
    # 🔧 FIXED: Current feeding status with proper timezone handling
    history_text += f"\n🍖 **CURRENT FEEDING STATUS:**\n"
    history_text += f"• Consecutive Detections: {feeding_tracker['consecutive_detections']}/5\n"
    history_text += f"• Detection History: {feeding_tracker['detection_history']}\n"
    
    if feeding_tracker['last_fed_time']:
        try:
            # Ensure both times are timezone-aware
            current_time = get_local_time()
            last_fed_time = feeding_tracker['last_fed_time']
            
            # Calculate time difference
            time_since_fed = current_time - last_fed_time
            hours_since = time_since_fed.total_seconds() / 3600
            
            history_text += f"• Last Fed: {format_local_time(last_fed_time)} ({hours_since:.1f}h ago)\n"
            remaining_hours = max(0, 4 - hours_since)
            history_text += f"• Auto-reset in: {remaining_hours:.1f} hours\n"
        except Exception as e:
            history_text += f"• Last Fed: Error calculating time ({e})\n"
    else:
        history_text += f"• Last Fed: Never\n"
    
    try:
        # Calculate session uptime with timezone-aware comparison
        current_time = get_local_time()
        session_start = feeding_tracker['session_start_time']
        session_uptime = current_time - session_start
        history_text += f"• Session Uptime: {session_uptime.total_seconds()/3600:.1f} hours\n"
    except Exception as e:
        history_text += f"• Session Uptime: Error calculating ({e})\n"
    
    return history_text

def clear_history():
    """Clear detection history and images"""
    global detection_history
    detection_history = []
    save_history()
    
    try:
        if os.path.exists(images_folder):
            for filename in os.listdir(images_folder):
                if filename.endswith('.jpg'):
                    os.remove(os.path.join(images_folder, filename))
    except Exception as e:
        print(f"⚠️ Error clearing images: {e}")
    
    return "🗑️ History and images cleared!", get_history_text(), []

def refresh_history():
    """Refresh history display"""
    return get_history_text(), get_history_images()

def reset_feeding_manually():
    """🍖 Manual feeding reset function"""
    global feeding_tracker
    
    try:
        feeding_tracker['consecutive_detections'] = 0
        feeding_tracker['detection_history'] = []
        feeding_tracker['last_fed_time'] = None
        
        if firebase_available:
            current_time = format_local_time()
            dog_ref = db.reference('DogCame')
            dog_ref.update({
                'Eaten': False,
                'Time': current_time
            })
        
        return f"🍖 Feeding tracker and Firebase reset successfully at {format_local_time()}!", get_history_text()
    except Exception as e:
        return f"❌ Error resetting feeding: {e}", get_history_text()

# Load existing history on startup
load_history()

# Create interface
with gr.Blocks(
    title="🐕👤🍖 Smart Pet & Human Detection with Feeding System",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1600px !important;
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🐕👤🍖 Smart Pet & Human Detection with Feeding System
    ### Powered by YOLOv8 AI Model + Complete Firebase Integration
    **🔥 Conditional Detection:** Always detects pets, detects humans only when Firebase IntruderAlert = True
    **🍖 Smart Feeding:** 5 consecutive dog detections → Auto-feed → 4h reset
    **🕐 Timezone:** Asia/Colombo (Local Time)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Upload Image")
            input_image = gr.Image(
                type="pil", 
                label="Upload Image for Detection",
                height=400
            )
            
            with gr.Row():
                detect_btn = gr.Button("🔍 Smart Detect", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            with gr.Row():
                reset_feeding_btn = gr.Button("🍖 Reset Feeding", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Detection Results")
            output_image = gr.Image(
                label="Detected Objects (with bounding boxes)",
                height=400
            )
            
            output_text = gr.Textbox(
                label="Detection Summary",
                lines=15,
                max_lines=25,
                show_copy_button=True
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Detection History & Feeding Statistics")
            
            history_text = gr.Textbox(
                label="📊 History, Statistics & Feeding Status",
                lines=25,
                max_lines=30,
                show_copy_button=True,
                value=get_history_text()  # 🔧 FIXED: This now works properly
            )
            
            with gr.Row():
                clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary")
                refresh_btn = gr.Button("🔄 Refresh History", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Image History Gallery")
            
            history_gallery = gr.Gallery(
                label="Recent Detections (with bounding boxes)",
                show_label=True,
                columns=2,
                rows=3,
                height=400,
                value=get_history_images()
            )
    
    gr.Markdown("### 📝 Try These Examples")
    gr.Examples(
        examples=[
            "https://images.unsplash.com/photo-1552053831-71594a27632d?w=600",  # Dog
            "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=600",  # Cat
            "https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=600",  # Person with dog
        ],
        inputs=input_image,
        outputs=[output_image, output_text, history_text, history_gallery],
        fn=predict,
        cache_examples=False
    )
    
    # Event handlers
    detect_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[output_image, output_text, history_text, history_gallery]
    )
    
    input_image.change(
        fn=predict,
        inputs=[input_image],
        outputs=[output_image, output_text, history_text, history_gallery]
    )
    
    clear_btn.click(
        fn=lambda: (None, "", get_history_text(), get_history_images()),
        outputs=[input_image, output_text, history_text, history_gallery]
    )
    
    clear_history_btn.click(
        fn=clear_history,
        outputs=[output_text, history_text, history_gallery]
    )
    
    refresh_btn.click(
        fn=refresh_history,
        outputs=[history_text, history_gallery]
    )
    
    reset_feeding_btn.click(
        fn=reset_feeding_manually,
        outputs=[output_text, history_text]
    )
    
    gr.Markdown(f"""
    ---
    **🤖 Model:** YOLOv8s | **🎯 Classes:** Dogs, Cats & Humans | **⚡ Confidence:** > 30%
    **🕐 Timezone:** {LOCAL_TIMEZONE} | **⏰ Current Time:** {format_local_time()}
    
    **🔥 Complete Firebase Integration:**
    - **Variables/IntruderAlert**: Controls human detection
    - **FeedCommand/FeedNow**: Controls feeding system
    - **DogCame/Number**: Auto-incremented with dog detections
    - **DogCame/Eaten**: Auto-set to True after 5 consecutive detections
    - **Intruder/IsHere**: Set when humans detected (if IntruderAlert=True)
    - **Auto-Reset**: 4-hour timer after feeding (timezone aware)
    - **Daily Reset**: All counters reset at 23:59 local time
    
    **🍖 Smart Feeding Logic:**
    - Track consecutive dog detections (last 5)
    - Trigger feeding after 5 consecutive detections
    - Auto-reset after 4 hours (local timezone)
    - Manual reset available
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )