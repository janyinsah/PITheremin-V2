# END OF YEAR PROJECT: THE PI THEREMIN (PI THEREMIN VERSION) 
# CREATED BY JOSIAH ANYINSAH-BONDZIE
# STUDENT ID: 8624637
#                                  DIFFERENCES
# - SAMPLESIZE DOWNSCALED
# - WEBCAM RESOLUTION DOWNSCALED
# - FRAME RATE CAPPED
# ---------------------------------------------------------------------------------
# IMPORT NECESSARY MODULES 
import mediapipe as mp # Module used to display and receive hand landmark data.
import cv2 # Used for capture video data for the Theremin.
import numpy as np # Maths module used to calculate waveform
import pygame # Game module, but used to continously produce sine wave on it's mixer channels.
import threading # Allow the program to make use of threading
import queue # First in First Out Data structure to import frequency and amplitude values into the sound playbacl thread
import time # Used to add intervals between blocks of code. 
# ---------------------------------------------------------------------------------
#                                       BODY OF CODE
# Create the pygame mixer.
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512) # Initialize the mixer with corresponding values passed as arguements.

print(pygame.mixer.get_init())

# Determine Frequency ranges for the PI Theremin, and resolution for cv2 frame.

MAIN_CHANNEL = pygame.mixer.Channel(0)

# n is defined as the moving average filter, this was an attempt to try and smove out the frequency as it produced arpeggiated sounds.
n = 10
# This numpy function takes the last average of n and assigns it to a variable called window.
window = np.ones(n)/n # Define a Hann window for signal procesing.

# Initialize mediapipe hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

lock_mixer = threading.Lock()

# Define and generate a sine wave through a function, using numpy.
def genSineWave(Freq, Amp):
    Freq = np.reshape(Freq, (-1, 1))
    SAMPLES = np.zeros((22050, 2), dtype=np.int16) # Initialize samples using a numpy array passing two channels as a tuple.
    SINE = np.sin(2 * np.pi * np.arange(22050) * Freq[0] / 22050)
    resizeAmp = np.resize(Amp, (22050,)) # Reshapes the Amplitude to fit within the sample set for the mixer to inteprate the array and play the generetated sound wave at the correct
    SCALED_WAVE = (SINE * 32767 * resizeAmp).astype(np.int16) # This scales the wave according to the size of the amplitude (moving amplitude changes of hand landmarks) and converts it to the dataype needed to process it as audio. Note that it is all the same datatype as the pygame mixer.
    with lock_mixer: # checks to make sure that the following code is only executed at one thread a time. (
        SAMPLES[:, 0] = SCALED_WAVE  # Index the samples array by 2 so that it matches the format of the steromixer.
        SAMPLES[:, 1] = SCALED_WAVE  # It duplicates the number of samples aqcross the stereo channel.
        SOUND = pygame.sndarray.make_sound(capGain(SAMPLES)) # Convert then generate the sine wave based of it's calculation of the position of the left/right hand co-ordinates.
        MAIN_CHANNEL.play(SOUND, loops = -1) # Plays the through the pygame channel.

def capGain(samples): # Function to appy to samples tor reduce clipping sound.
    maxAmp = np.iinfo(samples.dtype).max # Get the total amount of samples. 
    gain = min(1, (maxAmp / 2) / np.max(np.abs(samples)))  # Makes sure that the gain is never more than 1 using the min() function
    return (samples * gain).astype(samples.dtype) # Returns the result of the gain factor multipled by the samples in form of the original samples datatype.

# This is a thread safe function that determins audio playback through the pygame mixer
def play_sound(sound_queue):
    while True:
        sound_data = sound_queue.get() # Data from the sound queue is processed (the left hand and right hand values for frequency and amplitude)
        if sound_data is None: # If there is no data to be processed in the queue (the left and right hand features), no sound will be played.
            break 
        Freq, Amp = sound_data # Mapping the values received from the left hand and right hand co-ordinates to local variables within the thread-safe sound function.
        Amp = Amp.flatten()[:Freq.shape[0]] # Converts the 2 dimensional array which amplitude is stored to match the array length of frequency.
        genSineWave(Freq, Amp) # New frequency and values passed to generate a waveform based on the given values from the queue data structure.
        time.sleep(0.01) # Added a 1/10000 time break for processing reasons. (It just works better when it's there, but if it's too long the difference is to noticable)

camera = cv2.VideoCapture(1) #Created a videocapture object, that captures video from the default camera of the device.
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Sets the buffer size of the camera (video frame)  to 1. This is so the frame captures the most recent frame. (Real time fixes)
# Camera properties defined to reduce the resolution of the frame.
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320) # Define width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) # Define height
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Get the most recent frame.

frame_rate = 15  # The frame cap total of which the frames should not exceed per second.
delay = 1/frame_rate # Applies frame rate of how many frames appear per capture.


# Using Queue Data structures to ensure to seperate sound generation from the main thread (main thread deals with processing frames and hand landmark data)
sound_queue = queue.Queue() # Initialization
sound_thread = threading.Thread(target=play_sound, args=(sound_queue,)) # This data structure follows a first-in-first-out rule, which is necessary for processing sound in accordance to the most recent hand co-ordination.
sound_thread.start() # Start the data structure.

# These variables were taken from the mainthread and assigned outside of the loop as zero values. 
# This is to avoid crashes when there is 1 or no hands detected on the frame.
leftHandFreqSmoothed = np.zeros(1)
rightHandAmp = np.zeros(1)


hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.5) # Only maximum of two hands can be detected. This was previously inside the while loop which caused the memory leak.

while True: # Loop used to run infinitely. 
    while camera.isOpened(): # While the camera is on:
        ret, frame = camera.read() # Captures a single frame
        if not ret: # Break if you cant detect the webcam. 
            break
        frame = cv2.flip(frame, 1) # Flip the frame horizontally so that the the left and right hand positions are detected and mapped corrected.
        time.sleep(delay) # Frame rate cap for the Raspberry Pi Version
    # Like the old Thermein version, we draw hand landmarks onto the image by converting BGR to RGB and defining the number of hands for mediapipe to detect.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Colour is converted to RGB to be read by mediapipe as it uses RGB system for colours.
        hand_result = hands.process(frame) # Define the result of the hand detection and process the image onto the webcam stream. Only maximum of two hands can be detected
        # This is so that the Pi Theremin cannot be interfered by other hands.
#----------------------------------------------------------------
# Draw hands onto image (webcam), by converting RGB abck to BGR, as this is OpenCV's colour language and then draw the landmarks by detecting each of them using a for loop.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR) # Converts back to BGR to displayed on the opencv video frame. BGR is openCV's colour system.
        leftHand, rightHand = None, None # No detected hands on frame, initial hand values are set to None. 
        if hand_result.multi_hand_landmarks: # If hand landmarks are detected on the image:
            hand_landmark = hand_result.multi_hand_landmarks

            # Looks complicated, but actually isnt (List comprehension to make it a single line of code). We loop through EACH landmark on both hands and produce an average to return a 2 dimensional array of hand landmarks 
            hand_centers = [np.mean(np.array([[landmark.x, landmark.y] for landmark in hand.landmark]), axis=0) for hand in hand_landmark]  # It gets stored vertically across the array into two seperate parts. (axis=0)

            # Also looks devious but isn't. Again using list comprehension we merg the hand landmarks with the newly generated hand centered co-ordinates into one list of array.
            # The center hand landmarks are joined together in one list with it's corresponding hand landmark.
            sorted_hand_landmarks = [hand for _, hand in sorted(zip(hand_centers, hand_landmark), key=lambda pair: pair[0][0])] # Lambda is a one line function that pairs center hands and it's associated hand landmark and sorts them corresponding to the key[0][0] which returns the x co-ordinates (vertical axis 0 at index 0 [0][0]) from the most left x < 1 to the most right x > 1
            # We iterate through this sorted hand only taking the hand landmark and leaving the hand center landmarks

            leftHand = sorted_hand_landmarks[0] # Assign the left hand landmarks to the first set of hand landmarks identified form the key pair lambda functiom
            rightHand = sorted_hand_landmarks[1] if len(sorted_hand_landmarks) == 2 else None # Assigns the right hand the second index of hand landmarks only if the index length of the hand landmarks equates to 2. (Remember max_num_hands=2)

            for landmark_item in hand_result.multi_hand_landmarks: # Loop through each landmark in then landmark list: multi_hand_landmarks.
                mp_drawing.draw_landmarks(frame, landmark_item, connections=mp_hands.HAND_CONNECTIONS) # Draw the landmarks onto the image/frame. (Optional, only used to see how mediapipe works in response to the Theremin Program.)

            # This section of code gets the x and y axis for each landmark within the frame.
            leftHandCoord = np.array([[landmark.x, landmark.y] for landmark in leftHand.landmark])

            if rightHand is not None: # We check if the right hand is detected on the frame
                rightHandCoord = np.array([[landmark.x, landmark.y] for landmark in rightHand.landmark])  # Convert the right hand positions to a two dimensional numpy array
            else: 
                rightHandCoord = None # It doesn't exist if there is no right hand.

            if rightHandCoord is not None: # Again checking to see if the right hand is detected on the frame
                # minCoord and maxCoord is the result of calcualting the minimum and amximum co-ordinates of the left hand and right hand.
                # np.vstack is used to stack the co-ordinates into a 2d array in sequence vertically so it can be interpolated (next block of code).
                minCoord = np.min(np.vstack([leftHandCoord, rightHandCoord]), axis=0)
                maxCoord = np.max(np.vstack([leftHandCoord, rightHandCoord]), axis=0)
            else: 
                minCoord = np.min(leftHandCoord, axis=0) # Only detect and display landmarks of which represent the minimum and maximum values of the left hand
                maxCoord = np.max(leftHandCoord, axis=0)
                # In this case this would be NILL, and values that are sent would be zeros, this means no sound will be played.

            # Map distance of hands within frame, to freq and amp ranges for the Pi Theremin. 
            # Numpy interpolation so that even if the hand coordinations fall out of range (although not expected to due to vstack and the assignment of the coordinate arrays), that it is extrapolated back into the ranges each endpoint. e.g 100,1000.

            leftHandFreq = np.interp(leftHandCoord, [minCoord[0], maxCoord[0]], [100, 1000]) # Mapping the left hand co-ordinates within the range of set amp min, and amp max 0 and 1.
            # The [0] represents the x axis, which is what determines frequency for the left hand.
            # We flatten the left hand frequency to a 1D array so that it can be sequenced and convolved
            # Hann window used for signal processing and the same means that the input (leftHandFreq) signal will still match the output signal. This is passed as a parameter to genSineWave instead of the original left hand frequency.

            leftHandFreqSmoothed = np.convolve(leftHandFreq.flatten(), window, mode='same') 

            if rightHandCoord is not None:
                rightHandAmp = np.interp(rightHandCoord, [minCoord[1], maxCoord[1]], [0, 1])[:, np.newaxis] # np.newaxis adds an empty dimension so that numpy can interpolate rightHandAmps values to map it to the required range for amplitude of the wave
                # [1] a represents the Y axis, which is what will determine the amplitude for the right hand
            else:
                rightHandAmp = np.zeros_like(leftHandFreqSmoothed) # No sound will be played.

            # Generate a waveform using numpy, then send it to defined pygame channel above, this is so that the sound is constant throughout hand movement,
            sound_queue.put((leftHandFreqSmoothed, rightHandAmp))            
        else:
            sound_queue.put((np.zeros_like(leftHandFreqSmoothed), np.zeros_like(rightHandAmp))) # If there is no hands drawn on the frame, is() set the hand co-orindate values to 0 and pass it into the queue data structure.

        cv2.imshow("Pi Theremin", frame) # Display the frame, with given title "Pi Theremin."
        cv2.moveWindow("Pi Theremin", 0, 0) # Maps the frame to the top left of the screen. 
        if cv2.waitKey(1) == ord('q'): # Quit program and end while loop when q is pressed.
            break

    sound_queue.put(None) # Terminate the sound playback thread.
    sound_thread.join() # Wait for the sound thread to process the current amp and frequency values before playing back the next values.
    cv2.destroyAllWindows() # Terminate frame.