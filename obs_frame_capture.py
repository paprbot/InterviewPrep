from obswebsocket import obsws, requests
import cv2
import numpy as np
import time
import os
from datetime import datetime
import base64
from PIL import Image
import io
import json
from collections import defaultdict
import requests as http_requests
import mimetypes
import streamlit as st
import threading
import queue
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

class ImageUploader:
    def __init__(self):
        self.client_id = os.getenv('IMGUR_CLIENT_ID')
        if not self.client_id:
            raise ValueError("IMGUR_CLIENT_ID environment variable not set in .env file")
            
    def upload_image(self, image_path):
        """Upload an image to Imgur and return the HTTPS URL"""
        try:
            # Read the image file
            with open(image_path, 'rb') as image_file:
                # Prepare the request
                url = "https://api.imgur.com/3/image"
                headers = {
                    'Authorization': f'Client-ID {self.client_id}'
                }
                files = {
                    'image': image_file
                }
                
                # Upload to Imgur
                response = http_requests.post(url, headers=headers, files=files)
                
                if response.status_code == 200:
                    # Get the HTTPS URL from the response
                    data = response.json()['data']
                    return data['link']  # This is the HTTPS URL
                else:
                    print(f"Error uploading to Imgur: {response.status_code}")
                    print(response.text)
                    return None
                    
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None

class OBSCapture:
    def verify_api_key(self):
        """Verify that the API key is loaded and valid"""
        try:
            if self.use_gemini:
                # Verify Gemini API key by making a simple request
                response = self.gemini_model.generate_content("Test connection")
                print("Gemini API key is valid and loaded successfully")
                return True
            else:
                # Verify OpenAI API key
                models = self.openai_client.models.list()
                print("OpenAI API key is valid and loaded successfully")
                print(f"Available models: {len(models.data)}")
                return True
        except Exception as e:
            print(f"API key verification failed: {str(e)}")
            return False

    def __init__(self, use_gemini=False):
        self.output_dir = "captured_frames"
        self.unique_dir = "unique_frames"
        self.analysis_dir = "interview_analysis"
        self.test_images_dir = "test_images"  # Directory for test images
        self.fps = 1
        self.running = False
        self.stop_event = threading.Event()
        self.ws = None
        self.captured_files = []
        self.frame_hashes = {}
        self.duplicate_groups = defaultdict(list)
        self.unique_frames = []
        self.analysis_queue = queue.Queue()
        self.image_uploader = ImageUploader()
        self.use_gemini = use_gemini
        self.capture_thread = None
        self.analysis_in_progress = False
        self.test_mode = False
        self.api_initialized = False
        
        # Initialize AI client based on flag (only once)
        if not self.api_initialized:
            if self.use_gemini:
                # Initialize Gemini
                gemini_api_key = os.getenv('GEMINI_API_KEY')
                if not gemini_api_key:
                    raise ValueError("GEMINI_API_KEY environment variable not set")
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                print("Initialized Gemini API")
            else:
                # Initialize OpenAI
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self.openai_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.openai.com/v1"
                )
                print("Initialized OpenAI API")
            
            # Verify API key
            self.verify_api_key()
            self.api_initialized = True
        
        # Create output directories if they don't exist
        for directory in [self.output_dir, self.unique_dir, self.analysis_dir, self.test_images_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
            
    def calculate_frame_hash(self, frame_array):
        """Calculate a hash of the frame for comparison"""
        # Convert to grayscale and resize to reduce comparison complexity
        gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (32, 32))
        # Calculate mean and std of the resized image
        mean, std = cv2.meanStdDev(resized)
        # Create a hash using mean and std values
        return hash(tuple(mean.flatten()) + tuple(std.flatten()))
        
    def is_similar_frame(self, frame_array, threshold=0.95):
        """Check if the frame is similar to any previous frame"""
        frame_hash = self.calculate_frame_hash(frame_array)
        
        # If we've seen this exact hash before
        if frame_hash in self.frame_hashes:
            return True, frame_hash
            
        # Check for similar frames using structural similarity
        for existing_hash, existing_frame in self.frame_hashes.items():
            similarity = cv2.matchTemplate(
                cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(existing_frame, cv2.COLOR_RGB2GRAY),
                cv2.TM_CCOEFF_NORMED
            )[0][0]
            if similarity > threshold:
                return True, existing_hash
                
        return False, frame_hash
        
    def connect(self, host="192.168.1.2", port=4455, password="fNUnY0vismaURiWU"):
        """Connect to OBS WebSocket server"""
        try:
            self.ws = obsws(host, port, password)
            self.ws.connect()
            print(f"Connected to OBS at {host}:{port}")
            
            # Get OBS version info
            version = self.ws.call(requests.GetVersion())
            print("\nOBS Version Info:", json.dumps(version.datain, indent=2))
            
            # List all available scenes
            scenes = self.ws.call(requests.GetSceneList())
            print("\nAvailable scenes:")
            for scene in scenes.datain.get('scenes', []):
                print(f"- {scene['sceneName']}")
            
            return True
        except Exception as e:
            print(f"Failed to connect to OBS: {e}")
            return False
            
    def start_capture(self):
        """Start capturing frames"""
        if self.running:
            print("Capture is already running")
            return
            
        print("Starting capture...")
        self.running = True
        self.stop_event.clear()  # Clear the stop event
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="OBS_Capture_Thread")
        self.capture_thread.start()
        print(f"Started capture thread. Thread alive: {self.capture_thread.is_alive()}")

    def _capture_loop(self):
        """Internal capture loop"""
        last_capture_time = 0
        
        while self.running and not self.stop_event.is_set():
            current_time = time.time()
            
            # Check if it's time to capture a new frame (1 FPS)
            if current_time - last_capture_time >= 1.0:
                try:
                    # Check if we should still be running
                    if not self.running or self.stop_event.is_set():
                        print("Capture stopped during scene detection")
                        break
                        
                    # Get the current scene using GetCurrentProgramScene
                    try:
                        current_scene = self.ws.call(requests.GetCurrentProgramScene())
                        print("\nCurrent scene response:", json.dumps(current_scene.datain, indent=2))
                        
                        scene_name = current_scene.datain.get('currentProgramSceneName')
                        
                        if not scene_name:
                            print("No active scene found")
                            continue
                        
                        print(f"Capturing scene: {scene_name}")
                        
                        # Check if we should still be running
                        if not self.running or self.stop_event.is_set():
                            print("Capture stopped during scene processing")
                            break
                        
                        # Get scene sources
                        sources = self.ws.call(requests.GetSceneItemList(sceneName=scene_name))
                        print("Scene sources:", json.dumps(sources.datain, indent=2))
                    except Exception as e:
                        print(f"Error in OBS WebSocket call: {str(e)}")
                        if "disconnect" in str(e).lower() or "connection" in str(e).lower():
                            print("OBS connection lost - stopping capture")
                            break
                        continue
                    
                    # Find the Video Capture Device source
                    video_source = None
                    for item in sources.datain.get('sceneItems', []):
                        if item.get('sourceName') == 'Video Capture Device' and item.get('sceneItemEnabled', False):
                            video_source = item
                            break
                    
                    if not video_source:
                        print("No enabled Video Capture Device found in scene")
                        continue
                        
                    # Get the source dimensions from the transform
                    transform = video_source.get('sceneItemTransform', {})
                    width = transform.get('width', 1920)
                    height = transform.get('height', 1080)
                            
                    print(f"Attempting to capture Video Capture Device with dimensions: {width}x{height}")
                    
                    try:
                        # Check if we should still be running before taking screenshot
                        if not self.running or self.stop_event.is_set():
                            print("Capture stopped before taking screenshot")
                            break
                            
                        # First try GetSourceScreenshot
                        screenshot = self.ws.call(requests.GetSourceScreenshot(
                            sourceName='Video Capture Device',
                            imageFormat="jpg",
                            imageQuality=100,
                            imageWidth=width,
                            imageHeight=height
                        ))
                        
                        if not screenshot:
                            print("No response received from OBS WebSocket")
                            continue
                            
                        if not hasattr(screenshot, 'datain'):
                            print("Response missing datain attribute")
                            print("Raw response:", screenshot)
                            continue
                            
                        if not screenshot.datain:
                            print("Response datain is empty")
                            print("Raw response:", screenshot)
                            continue
                            
                        # Check for either img or imageData key
                        image_data = None
                        if 'img' in screenshot.datain:
                            image_data = screenshot.datain['img']
                        elif 'imageData' in screenshot.datain:
                            # Remove the data:image/jpg;base64, prefix if present
                            image_data = screenshot.datain['imageData']
                            if ',' in image_data:
                                image_data = image_data.split(',', 1)[1]
                        else:
                            print("Response missing image data")
                            print("Response data:", json.dumps(screenshot.datain, indent=2))
                            continue
                            
                        # Convert base64 image to numpy array
                        try:
                            img_data = base64.b64decode(image_data)
                            img = Image.open(io.BytesIO(img_data))
                            frame_array = np.array(img)
                            
                            # Save the frame
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(self.output_dir, f"frame_{timestamp}.jpg")
                            cv2.imwrite(filename, cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))
                            self.captured_files.append(filename)
                            
                            # Check for duplicates
                            is_duplicate, frame_hash = self.is_similar_frame(frame_array)
                            if is_duplicate:
                                self.duplicate_groups[frame_hash].append(filename)
                            else:
                                self.frame_hashes[frame_hash] = frame_array
                                self.duplicate_groups[frame_hash] = [filename]
                                # Save unique frame to unique_frames directory
                                unique_filename = os.path.join(self.unique_dir, f"unique_{timestamp}.jpg")
                                cv2.imwrite(unique_filename, cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))
                                self.unique_frames.append(unique_filename)
                            
                            print(f"Successfully captured frame: {filename}")
                        except Exception as e:
                            print(f"Error processing image data: {str(e)}")
                            print("Image data length:", len(image_data) if image_data else 0)
                            
                    except Exception as e:
                        print(f"Error taking screenshot: {str(e)}")
                        print("Error type:", e.__class__.__name__)
                        print("Full error details:", str(e))
                        
                        # Try alternative method if first attempt fails
                        try:
                            print("\nTrying alternative screenshot method...")
                            alt_screenshot = self.ws.call(requests.TakeSourceScreenshot(
                                sourceName='Video Capture Device',
                                imageFormat="jpg",
                                imageQuality=100
                            ))
                            print("Alternative method response:", json.dumps(alt_screenshot.datain, indent=2) if alt_screenshot and alt_screenshot.datain else "No response data")
                        except Exception as e2:
                            print(f"Alternative method failed: {str(e2)}")
                except Exception as e:
                    print(f"Error capturing frame: {e}")
                
                last_capture_time = current_time
            
            # Check running flag more frequently
            for _ in range(20):  # Check every 0.005 seconds for faster response
                if not self.running or self.stop_event.is_set():
                    break
                time.sleep(0.005)
        
        print("Capture loop exited - self.running is now False")
        
        # Always trigger analysis when capture loop exits
        if self.captured_files and not self.analysis_in_progress:
            print("Capture loop exited - triggering analysis automatically")
            self.analysis_in_progress = True
            self.prepare_gpt4_analysis()
            self.analysis_in_progress = False

    def load_test_images(self, folder_path, max_images=10):
        """Load test images from a folder for analysis"""
        # Prevent repeated calls - check if we already have test images loaded
        if hasattr(self, '_loading_test_images') and self._loading_test_images:
            print("Test images already being loaded, skipping...")
            return True
        
        # Check if we already have test images loaded from this folder
        if (self.test_mode and self.unique_frames and 
            any('test_image_' in os.path.basename(f) for f in self.unique_frames)):
            print("Test images already loaded, skipping...")
            return True
        
        self._loading_test_images = True
        print(f"Loading test images from: {folder_path}")
        self.test_mode = True
        self.captured_files = []
        self.unique_frames = []
        self.frame_hashes = {}
        self.duplicate_groups = defaultdict(list)
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Get all image files from the folder
        image_files = []
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(folder_path, file))
        
        # Sort files by name and limit to max_images
        image_files.sort()
        image_files = image_files[:max_images]
        
        if not image_files:
            print(f"No image files found in {folder_path}")
            return False
        
        print(f"Found {len(image_files)} image files")
        
        # Copy images to unique_frames directory for analysis
        for i, image_path in enumerate(image_files):
            try:
                # Copy to unique_frames with a new name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = os.path.join(self.unique_dir, f"test_image_{i+1}_{timestamp}.jpg")
                
                # Convert to JPEG if needed
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(new_filename, 'JPEG')
                
                self.unique_frames.append(new_filename)
                self.captured_files.append(new_filename)
                
                print(f"Loaded test image: {os.path.basename(image_path)} -> {os.path.basename(new_filename)}")
                
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(self.unique_frames)} test images")
        
        # Trigger analysis automatically for test mode
        if len(self.unique_frames) > 0 and not self.analysis_in_progress:
            print("Triggering analysis for test images...")
            self.analysis_in_progress = True
            self.prepare_gpt4_analysis()
            self.analysis_in_progress = False
        
        # Reset loading flag
        self._loading_test_images = False
        return len(self.unique_frames) > 0
        
        # Force thread termination
        import threading
        current_thread = threading.current_thread()
        if current_thread.is_alive():
            print(f"Thread {current_thread.name} is still alive after loop exit - forcing termination")
            # Force the thread to exit by setting a flag that will be checked
            self.running = False
            self.stop_event.set()
        else:
            print(f"Thread {current_thread.name} terminated normally")

    def check_openai_quota(self):
        """Check OpenAI API quota and status"""
        try:
            # Get subscription info
            subscription = self.openai_client.billing.subscription()
            
            # Get usage info
            usage = self.openai_client.billing.usage(
                start_date=datetime.now().strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            # Print status information
            print("\nOpenAI API Status:")
            print(f"Plan: {subscription.plan.title}")
            print(f"Status: {subscription.status}")
            print(f"Today's usage: ${usage.total_usage / 100:.2f}")
            print(f"Hard limit: ${subscription.hard_limit_usd:.2f}")
            
            # Check if we're close to the limit
            if usage.total_usage >= subscription.hard_limit_usd * 100 * 0.9:  # 90% of limit
                warning_msg = "Warning: You are close to your API usage limit!"
                print(warning_msg)
                self.analysis_queue.put(warning_msg)
                return False
                
            return True
            
        except Exception as e:
            error_msg = f"Error checking OpenAI quota: {str(e)}"
            print(error_msg)
            self.analysis_queue.put(error_msg)
            return False

    def prepare_gpt4_analysis(self):
        """Send unique frames to AI API for analysis"""
        print("=== PREPARE_GPT4_ANALYSIS METHOD CALLED ===")
        if not self.unique_frames:
            print("No unique frames to analyze")
            return

        # Read the prompt from file
        try:
            with open('prompt.txt', 'r', encoding='utf-8') as f:
                prompt = f.read()
        except Exception as e:
            print(f"Error reading prompt file: {str(e)}")
            return

        # Apply coding language modifications
        coding_language = getattr(st.session_state, 'coding_language', 'Both')
        if coding_language == "Python":
            prompt = prompt.replace("C++", "Python")
        elif coding_language == "C++":
            prompt = prompt.replace("Python", "C++")

        try:
            start_time = time.time()
            
            if self.use_gemini:
                # Prepare images for Gemini
                images = []
                useCapturedFrames = True
                if useCapturedFrames:
                    frame_list = self.captured_files
                else:
                    frame_list = self.unique_frames
                
                for frame_path in frame_list:
                    try:
                        if not os.path.exists(frame_path):
                            print(f"Warning: File not found: {frame_path}")
                            continue
                        image = Image.open(frame_path)
                        images.append(image)
                    except Exception as e:
                        print(f"Error loading image {frame_path}: {str(e)}")
                        continue
                
                if not images:
                    print("No valid images to analyze")
                    return
                
                # Make Gemini API call
                response = self.gemini_model.generate_content([prompt, *images])
                analysis = response.text
            else:
                # Prepare messages for OpenAI
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # Add images to OpenAI request
                valid_images = 0
                for frame_path in self.unique_frames:
                    try:
                        if not os.path.exists(frame_path):
                            print(f"Warning: File not found: {frame_path}")
                            continue
                            
                        with open(frame_path, 'rb') as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            messages[0]["content"].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            })
                            valid_images += 1
                    except Exception as e:
                        print(f"Error processing image {frame_path}: {str(e)}")
                        continue
                
                if valid_images == 0:
                    print("No valid images to analyze")
                    return
                
                # Make OpenAI API call
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=4096
                )
                analysis = response.choices[0].message.content
            
            stop_time = time.time()
            print(f"Time taken: {stop_time - start_time} seconds")
            
            # Save the analysis
            analysis_file = os.path.join(self.analysis_dir, "interview_analysis.txt")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(analysis)
            
            # Update states directly
            if 'can_stop' in st.session_state:
                st.session_state.can_stop = False
                st.session_state.capturing = False
                st.session_state.analysis_complete = True
                st.session_state.can_start = True
                st.session_state.running = False  # Ensure capture is stopped
            
            # Put the analysis in the queue for Streamlit
            self.analysis_queue.put({
                'type': 'analysis',
                'content': analysis
            })
                
            print(f"\nAnalysis completed and saved to: {analysis_file}")
            print("\nAnalysis summary:")
            print(analysis[:500] + "..." if len(analysis) > 500 else analysis)
                
        except Exception as e:
            error_msg = f"Error during API call: {str(e)}"
            print(error_msg)
            
            # Update states directly on error
            if 'can_stop' in st.session_state:
                st.session_state.can_stop = False
                st.session_state.capturing = False
                st.session_state.analysis_complete = True
                st.session_state.can_start = True
                st.session_state.running = False  # Ensure capture is stopped
            
            self.analysis_queue.put({
                'type': 'error',
                'content': error_msg
            })

    def stop_capture(self):
        """Stop capturing frames"""
        print("=== STOP_CAPTURE METHOD CALLED ===")
        if not self.running:
            print("Capture is not running")
            return
            
        print("Stopping capture...")
        self.running = False
        self.stop_event.set()  # Set the stop event to signal the thread to stop
        
        # Force disconnect from OBS to interrupt any blocking calls
        if self.ws:
            try:
                print("Disconnecting from OBS to interrupt blocking calls...")
                self.ws.disconnect()
            except Exception as e:
                print(f"Error disconnecting from OBS: {str(e)}")
        
        # Don't wait for thread - just clear the reference immediately
        if self.capture_thread:
            print("Clearing capture thread reference...")
            self.capture_thread = None
        
        # Force clear thread reference regardless
        self.capture_thread = None
        
        # Ensure the thread is completely stopped
        if hasattr(self, 'capture_thread') and self.capture_thread:
            self.capture_thread = None
            
        # Force clear the running flag
        self.running = False
        
        if self.ws:
            try:
                self.ws.disconnect()
            except Exception as e:
                print(f"Error disconnecting from OBS: {str(e)}")
            
        # Print summary of captured files
        if self.captured_files:
            print("\nCaptured frames summary:")
            print(f"Total frames captured: {len(self.captured_files)}")
            
            # Analyze duplicates
            unique_frames = len(self.frame_hashes)
            duplicate_frames = len(self.captured_files) - unique_frames
            
            print(f"\nFrame Analysis:")
            print(f"Unique frames: {unique_frames}")
            print(f"Duplicate frames: {duplicate_frames}")
            print(f"Duplicate percentage: {(duplicate_frames/len(self.captured_files))*100:.1f}%")
            
            if duplicate_frames > 0:
                print("\nDuplicate groups:")
                for hash_val, files in self.duplicate_groups.items():
                    if len(files) > 1:
                        print(f"\nGroup with {len(files)} identical frames:")
                        for file in files:
                            print(f"- {file}")
            
            print("\nAll captured files:")
            for filename in self.captured_files:
                print(f"- {filename}")
                
            print("\nUnique frames saved:")
            for filename in self.unique_frames:
                print(f"- {filename}")
                
            # Prepare GPT-4 analysis (only if not already triggered by capture loop)
            print(f"Preparing analysis with {len(self.unique_frames)} unique frames...")
            if not self.analysis_in_progress and self.captured_files:
                self.analysis_in_progress = True
                print("Starting GPT-4 analysis from stop_capture method...")
                self.prepare_gpt4_analysis()
                self.analysis_in_progress = False
                print("Analysis preparation completed")
            else:
                print("Analysis already in progress or no frames captured, skipping...")
        else:
            print("\nNo frames were captured during this session.")
            
        # Clear capture-related variables
        self.captured_files = []
        self.frame_hashes = {}
        self.duplicate_groups = defaultdict(list)
        self.unique_frames = []

def get_available_models():
    """Query available OpenAI models"""
    try:
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        response = http_requests.get(
            "https://api.openai.com/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            models = response.json()["data"]
            return [model["id"] for model in models]
        else:
            print(f"Error querying models: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Error getting models: {str(e)}")
        return []

def run_streamlit():
    """Run the Streamlit interface"""
    st.set_page_config(page_title="Interview Analysis", layout="wide")
    
    st.title("Coding Interview Analysis")
    
    # Initialize session state
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'capture' not in st.session_state:
        st.session_state.capture = None
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'capturing' not in st.session_state:
        st.session_state.capturing = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'can_start' not in st.session_state:
        st.session_state.can_start = True
    if 'can_stop' not in st.session_state:
        st.session_state.can_stop = False
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'coding_language' not in st.session_state:
        st.session_state.coding_language = 'Both'
    
    # Function to update button states
    def update_button_states():
        if st.session_state.analysis_complete:
            st.session_state.can_start = True
            st.session_state.can_stop = False
            st.session_state.capturing = False
            st.session_state.running = False
        elif st.session_state.capturing:
            st.session_state.can_start = False
            st.session_state.can_stop = True
        else:
            st.session_state.can_start = True
            st.session_state.can_stop = False
    
    # Create the capture instance if not exists
    if st.session_state.capture is None:
        st.session_state.capture = OBSCapture(use_gemini=True)
    else:
        # Clean up any orphaned threads on app restart
        if (st.session_state.capture.capture_thread and 
            st.session_state.capture.capture_thread.is_alive() and 
            not st.session_state.capturing):
            print("Cleaning up orphaned capture thread...")
            st.session_state.capture.capture_thread = None
            st.session_state.capture.running = False
    
    # Add API selection in sidebar
    with st.sidebar:
        st.header("API Settings")
        use_gemini = st.checkbox("Use Gemini API instead of OpenAI", value=True)
        
        # Only create new instance if it doesn't exist or if API type actually changed
        if st.session_state.capture is None:
            st.session_state.capture = OBSCapture(use_gemini=use_gemini)
        elif st.session_state.capture.use_gemini != use_gemini:
            # Only recreate if API type actually changed
            print(f"Switching from {'Gemini' if st.session_state.capture.use_gemini else 'OpenAI'} to {'Gemini' if use_gemini else 'OpenAI'}")
            st.session_state.capture = OBSCapture(use_gemini=use_gemini)
    
    # Add coding language selection
    with st.sidebar:
        st.header("Coding Language")
        coding_language = st.selectbox(
            "Select the programming language for analysis:",
            ["Python", "C++", "Both"],
            help="Choose the programming language(s) to focus on during the interview analysis"
        )
        
        # Store the selected language in session state
        st.session_state.coding_language = coding_language
    
    # Add test mode section
    with st.sidebar:
        st.header("Test Mode")
        test_mode = st.checkbox("Enable Test Mode", help="Use test images instead of live capture")
        
        if test_mode:
            st.info("Test mode enabled - will use images from folder instead of live capture")
            
            # Test image folder selection
            # set default C:\projectalpha\captured_frames
            test_folder = st.text_input(
                "Test Images Folder Path:", 
                value="C:\projectalpha\captured_frames",
                help="Path to folder containing test images"
            )
            
            # Number of images to use
            max_test_images = st.slider(
                "Number of Images to Use:", 
                min_value=1, 
                max_value=20, 
                value=5,
                help="Maximum number of images to load from the folder"
            )
            
            # Load test images button
            if 'test_images_loaded' not in st.session_state:
                st.session_state.test_images_loaded = False
            if 'test_images_folder' not in st.session_state:
                st.session_state.test_images_folder = ""
            
            # Check if we've already loaded images from this folder
            already_loaded = (st.session_state.test_images_loaded and 
                            st.session_state.test_images_folder == test_folder)
            
            if st.button("Load Test Images", disabled=already_loaded):
                if st.session_state.capture:
                    with st.spinner("Loading test images and starting analysis..."):
                        success = st.session_state.capture.load_test_images(test_folder, max_test_images)
                        if success:
                            st.session_state.test_images_loaded = True
                            st.session_state.test_images_folder = test_folder
                            st.success(f"Loaded {len(st.session_state.capture.unique_frames)} test images and started analysis!")
                            st.session_state.capturing = False
                            st.session_state.analysis_complete = False
                            st.session_state.can_start = True
                            st.session_state.can_stop = False
                            # Don't rerun here - let the analysis complete naturally
                        else:
                            st.error("Failed to load test images. Check the folder path.")
            
            # Show loaded test images
            if st.session_state.capture and st.session_state.capture.unique_frames:
                st.write(f"**Loaded {len(st.session_state.capture.unique_frames)} test images:**")
                for i, frame_path in enumerate(st.session_state.capture.unique_frames):
                    st.write(f"{i+1}. {os.path.basename(frame_path)}")
                
                # Analyze test images button
                if st.button("Analyze Test Images"):
                    if st.session_state.capture and not st.session_state.capture.analysis_in_progress:
                        st.session_state.capture.analysis_in_progress = True
                        st.session_state.capture.prepare_gpt4_analysis()
                        st.session_state.capture.analysis_in_progress = False
                        st.success("Analysis started!")
                        st.rerun()
                
                # Reset button to allow loading new test images
                if st.button("Reset Test Images"):
                    st.session_state.test_images_loaded = False
                    st.session_state.test_images_folder = ""
                    if st.session_state.capture:
                        st.session_state.capture.captured_files = []
                        st.session_state.capture.unique_frames = []
                        st.session_state.capture.test_mode = False
                        st.session_state.capture._loading_test_images = False
                    st.success("Test images reset. You can load new images now.")
                    st.rerun()
        else:
            # Clear test mode data when disabled
            if st.session_state.capture and st.session_state.capture.test_mode:
                st.session_state.capture.test_mode = False
                st.session_state.capture.captured_files = []
                st.session_state.capture.unique_frames = []
                st.info("Test mode disabled - cleared test images")
    
    # Add connection form
    with st.sidebar:
        st.header("OBS Connection Settings")
        host = st.text_input("Host", value=(os.getenv("OBS_HOST") or "192.168.1.7"))
        port = st.number_input("Port", value=(os.getenv("OBS_PORT") or 4455))
        password = st.text_input("Password", value=(os.getenv("OBS_PASSWORD") or "fNUnY0vismaURiWU"), type="password")
        
        if st.button("Connect to OBS"):
            if st.session_state.capture.connect(host, port, password):
                st.session_state.connected = True
                st.success("Connected to OBS!")
            else:
                st.session_state.connected = False
                st.error("Failed to connect to OBS")
    
    # Create columns for start/stop buttons
    col1, col2, col3 = st.columns(3)
    
    # Add status indicator
    thread_running = (st.session_state.capture and 
                     st.session_state.capture.capture_thread and 
                     st.session_state.capture.capture_thread.is_alive())
    
    analysis_in_progress = (st.session_state.capture and 
                           st.session_state.capture.analysis_in_progress)
    
    if analysis_in_progress:
        st.info("🔄 Analysis in progress...")
    elif st.session_state.capturing and thread_running:
        st.success("🟢 Capture is running...")
    elif st.session_state.capturing and not thread_running:
        st.warning("🟡 Capture should be running but thread is not alive")
    elif not st.session_state.capturing and thread_running:
        st.error("🔴 Capture should be stopped but thread is still alive!")
    else:
        st.info("⚪ Capture is stopped")
    
    # Debug information
    if st.session_state.capture:
        with st.expander("Debug Info"):
            st.write(f"Session capturing: {st.session_state.capturing}")
            st.write(f"Session running: {st.session_state.running}")
            st.write(f"Capture running: {st.session_state.capture.running}")
            st.write(f"Stop event set: {st.session_state.capture.stop_event.is_set()}")
            st.write(f"Thread exists: {st.session_state.capture.capture_thread is not None}")
            if st.session_state.capture.capture_thread:
                st.write(f"Thread alive: {st.session_state.capture.capture_thread.is_alive()}")
            st.write(f"OBS connected: {st.session_state.capture.ws is not None}")
    
    # Start capture button
    with col1:
        start_tooltip = "Start capturing frames from OBS. Make sure you're connected to OBS first!"
        if st.session_state.analysis_complete:
            start_tooltip = "Start a new capture session. Previous analysis will be cleared."
        elif not st.session_state.connected:
            start_tooltip = "Please connect to OBS first before starting capture."
        
        # Disable in test mode
        button_disabled = not st.session_state.can_start or test_mode
            
        if st.button("Start Capture", 
                    disabled=button_disabled,
                    help=start_tooltip if not test_mode else "Disabled in test mode"):
            if test_mode:
                st.warning("Live capture is disabled in test mode. Use 'Load Test Images' instead.")
                return
            if not st.session_state.connected:
                st.error("Please connect to OBS first!")
                return
                
            st.session_state.capturing = True
            st.session_state.analysis_complete = False
            st.session_state.running = True
            update_button_states()
            # Start capture in a separate thread
            capture_thread = threading.Thread(target=st.session_state.capture.start_capture)
            capture_thread.start()
    
    # Stop capture button
    with col2:
        stop_tooltip = "Stop capturing frames and start analysis"
        if not st.session_state.capturing:
            stop_tooltip = "Capture must be running to stop"
        
        # Disable in test mode
        stop_button_disabled = not st.session_state.can_stop or st.session_state.analysis_complete or test_mode
            
        if st.button("Stop Capture", 
                    disabled=stop_button_disabled,
                    help=stop_tooltip if not test_mode else "Disabled in test mode"):
            if st.session_state.capture:
                # Stop capture first - this will set the running flag to False
                st.session_state.capture.stop_capture()
                
                # Update session state after stopping
                st.session_state.can_stop = False
                st.session_state.capturing = False
                st.session_state.analysis_complete = True
                st.session_state.running = False
                st.session_state.can_start = True
                
                st.success("Capture stopped and analysis complete!")
                st.rerun()  # Force a rerun to update button states
    
    # Force stop button (emergency stop)
    with col3:
        if st.button("🛑 Force Stop", 
                    help="Emergency stop - forcefully terminates capture if normal stop doesn't work"):
            if st.session_state.capture:
                print("Force stopping capture...")
                # Disconnect from OBS immediately
                if st.session_state.capture.ws:
                    try:
                        st.session_state.capture.ws.disconnect()
                    except:
                        pass
                
                # Set all stop flags
                st.session_state.capture.running = False
                st.session_state.capture.stop_event.set()
                
                # Force clear the thread - this is the key fix
                if st.session_state.capture.capture_thread:
                    print("Force clearing capture thread...")
                    st.session_state.capture.capture_thread = None
                
                # Update session state
                st.session_state.capturing = False
                st.session_state.analysis_complete = True
                st.session_state.can_start = True
                st.session_state.can_stop = False
                st.session_state.running = False
                
                st.error("Capture force stopped!")
                st.rerun()
    
    # Kill thread button (for persistent threads)
    if (st.session_state.capture and 
        st.session_state.capture.capture_thread and 
        st.session_state.capture.capture_thread.is_alive() and 
        not st.session_state.capturing):
        if st.button("💀 Kill Orphaned Thread", 
                    help="Kill a thread that's still running even though capture should be stopped"):
            print("Killing orphaned capture thread...")
            st.session_state.capture.capture_thread = None
            st.session_state.capture.running = False
            st.success("Orphaned thread killed!")
            st.rerun()
    
    # Create containers for analysis and images
    analysis_container = st.container()
    images_container = st.container()
    
    # Monitor the analysis queue and display frames
    try:
        # Check for new analysis results
        try:
            queue_item = st.session_state.capture.analysis_queue.get_nowait()
            if isinstance(queue_item, dict):
                # Handle state updates from the queue
                if 'state_update' in queue_item:
                    for key, value in queue_item['state_update'].items():
                        st.session_state[key] = value
                    st.session_state.analysis_complete = True
                    st.session_state.can_stop = False  # Ensure stop button is disabled
                    st.session_state.can_start = True  # Ensure start button is enabled
                
                # Handle content
                if queue_item['type'] == 'analysis':
                    st.session_state.analysis = queue_item['content']
                elif queue_item['type'] == 'error':
                    st.error(queue_item['content'])
            else:
                # Handle legacy string messages
                st.session_state.analysis = queue_item
                st.session_state.analysis_complete = True
                st.session_state.can_stop = False  # Ensure stop button is disabled
                st.session_state.can_start = True  # Ensure start button is enabled
        except queue.Empty:
            pass
        
        # Display analysis results
        with analysis_container:
            if st.session_state.analysis:
                st.markdown("## Analysis")
                st.markdown(st.session_state.analysis)
        
        # Display captured frames (only once, not in a loop)
        if st.session_state.capture and st.session_state.capture.unique_frames:
            with images_container:
                st.markdown("## Captured Frames")
                cols = st.columns(min(3, len(st.session_state.capture.unique_frames)))
                for i, frame_path in enumerate(st.session_state.capture.unique_frames):
                    with cols[i % 3]:
                        st.image(frame_path, caption=f"Frame {i+1}")
            
    except Exception as e:
        st.error(f"Error in display: {str(e)}")
        st.session_state.analysis_complete = True
        update_button_states()
    
    # Add a button to exit the application
    if st.sidebar.button("Exit Application"):
        if st.session_state.capture:
            st.session_state.capture.stop_capture()
        st.stop()

def main():
    run_streamlit()

if __name__ == "__main__":
    main() 