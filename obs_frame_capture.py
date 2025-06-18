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
        self.fps = 1
        self.running = False
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
        
        # Initialize AI client based on flag
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
        
        # Create output directories if they don't exist
        for directory in [self.output_dir, self.unique_dir, self.analysis_dir]:
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
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()
        print("Started capture thread")

    def _capture_loop(self):
        """Internal capture loop"""
        last_capture_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Check if it's time to capture a new frame (1 FPS)
            if current_time - last_capture_time >= 1.0:
                try:
                    # Get the current scene using GetCurrentProgramScene
                    current_scene = self.ws.call(requests.GetCurrentProgramScene())
                    print("\nCurrent scene response:", json.dumps(current_scene.datain, indent=2))
                    
                    scene_name = current_scene.datain.get('currentProgramSceneName')
                    
                    if not scene_name:
                        print("No active scene found")
                        continue
                    
                    print(f"Capturing scene: {scene_name}")
                    
                    # Get scene sources
                    sources = self.ws.call(requests.GetSceneItemList(sceneName=scene_name))
                    print("Scene sources:", json.dumps(sources.datain, indent=2))
                    
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
            
            time.sleep(0.1)  # Small delay to prevent high CPU usage

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

        try:
            start_time = time.time()
            
            if self.use_gemini:
                # Prepare images for Gemini
                images = []
                for frame_path in self.unique_frames:
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
        if not self.running:
            print("Capture is not running")
            return
            
        print("Stopping capture...")
        self.running = False
        
        # Update states immediately
        if 'can_stop' in st.session_state:
            st.session_state.can_stop = False
            st.session_state.capturing = False
            st.session_state.analysis_complete = True
            st.session_state.can_start = True
        
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
            if self.capture_thread.is_alive():
                print("Warning: Capture thread did not stop gracefully")
                # Force stop the thread if it's still running
                self.capture_thread = None
        
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
                
            # Prepare GPT-4 analysis
            if not self.analysis_in_progress:
                self.analysis_in_progress = True
                self.prepare_gpt4_analysis()
                self.analysis_in_progress = False
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
    
    # Add API selection in sidebar
    with st.sidebar:
        st.header("API Settings")
        use_gemini = st.checkbox("Use Gemini API instead of OpenAI", value=True)
        
        if st.session_state.capture is None or st.session_state.capture.use_gemini != use_gemini:
            st.session_state.capture = OBSCapture(use_gemini=use_gemini)
    
    # Add connection form
    with st.sidebar:
        st.header("OBS Connection Settings")
        host = st.text_input("Host", value="192.168.1.2")
        port = st.number_input("Port", value=4455)
        password = st.text_input("Password", value="fNUnY0vismaURiWU", type="password")
        
        if st.button("Connect to OBS"):
            if st.session_state.capture.connect(host, port, password):
                st.session_state.connected = True
                st.success("Connected to OBS!")
            else:
                st.session_state.connected = False
                st.error("Failed to connect to OBS")
    
    # Create columns for start/stop buttons
    col1, col2 = st.columns(2)
    
    # Start capture button
    with col1:
        start_tooltip = "Start capturing frames from OBS. Make sure you're connected to OBS first!"
        if st.session_state.analysis_complete:
            start_tooltip = "Start a new capture session. Previous analysis will be cleared."
        elif not st.session_state.connected:
            start_tooltip = "Please connect to OBS first before starting capture."
            
        if st.button("Start Capture", 
                    disabled=not st.session_state.can_start,
                    help=start_tooltip):
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
            
        if st.button("Stop Capture", 
                    disabled=not st.session_state.can_stop or st.session_state.analysis_complete,
                    help=stop_tooltip):
            if st.session_state.capture:
                # Disable stop button immediately
                st.session_state.can_stop = False
                st.session_state.capturing = False
                st.session_state.analysis_complete = True
                st.session_state.running = False
                st.session_state.can_start = True
                
                # Stop capture and start analysis
                st.session_state.capture.stop_capture()
                st.success("Capture stopped and analysis complete!")
                st.rerun()  # Force a rerun to update button states
    
    # Create containers for analysis and images
    analysis_container = st.container()
    images_container = st.container()
    
    # Monitor the analysis queue and display frames
    try:
        while True:
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
                
                with analysis_container:
                    if st.session_state.analysis:
                        st.markdown("## Analysis")
                        st.markdown(st.session_state.analysis)
            except queue.Empty:
                pass
            
            # Display captured frames
            if st.session_state.capture.unique_frames:
                with images_container:
                    st.markdown("## Captured Frames")
                    cols = st.columns(min(3, len(st.session_state.capture.unique_frames)))
                    for i, frame_path in enumerate(st.session_state.capture.unique_frames):
                        with cols[i % 3]:
                            st.image(frame_path, caption=f"Frame {i+1}")
            
            # Add a small delay to prevent high CPU usage
            time.sleep(0.1)
            
    except Exception as e:
        st.error(f"Error in display loop: {str(e)}")
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