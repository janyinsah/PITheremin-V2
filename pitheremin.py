# END OF YEAR PROJECT: THE PI THEREMIN
# CREATED BY JOSIAH ANYINSAH-BONDZIE
# STUDENT ID: 8624637
# ---------------------------------------------------------------------------------
# IMPORT NECESSARY MODULES 
import mediapipe as mp # Module used to display and receive hand landmark data.
import cv2 # Used for capture video data for the Theremin.
import numpy as np # Maths module used to calculate waveform
import pygame # Game module, but used to continously produce sine wave on it's mixer channels.
# ---------------------------------------------------------------------------------
#                                       BODY OF CODE
# Create the pygame mixer.
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512) # Initialize the mixer with corresponding values passed as arguements.

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

# Define and generate a sine wave through a function, using numpy.
def genSineWave(Freq, Amp):
    Freq = np.reshape(Freq, (-1, 1))
    SAMPLES = np.zeros((44100, 2), dtype=np.int16) # Initialize samples using a numpy array passing two channels as a tuple.
    SINE = np.sin(2 * np.pi * np.arange(44100) * Freq[0]/44100)
    SCALED_WAVE = (SINE * 32767 * Amp[1]/2).astype(np.int16)
    SAMPLES[:, 0] = SCALED_WAVE  # Index the samples array by 2 so that it matches the format of the steromixer.
    SAMPLES[:, 1] = SCALED_WAVE  # It duplicates the number of samples across the stereo channel 
    SOUND = pygame.sndarray.make_sound(capGain(SAMPLES)) # Convert then generate the sine wave based of it's calculation of the position of the left/right hand co-ordinates.
    MAIN_CHANNEL.play(SOUND, loops = -1) # Plays the through the pygame channel.

def capGain(samples): # Function to appy to samples tor reduce clipping sound.
    maxAmp = np.iinfo(samples.dtype).max
    gain = min(1, (maxAmp / 2) / np.max(np.abs(samples))) 
    return (samples * gain).astype(samples.dtype)


frame = cv2.VideoCapture(0) # Open up a frame, for webcam streaming.

while True: # Loop used to run infinitely. 
    camera = cv2.VideoCapture(0) #Created a videocapture object, that captures video from the default camera of the device.
    while camera.isOpened(): # While the camera is on:
        ret, frame = camera.read() # Captures a single frame

    # Like the old Thermein version, we draw hand landmarks onto the image by converting BGR to RGB and defining the number of hands for mediapipe to detect.
        frame.flags.writeable = False # 
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # Colour is converted to RGB to be read by mediapipe as it uses RGB system for colours.
        hand_result = mp_hands.Hands(max_num_hands=2).process(frame) # Define the result of the hand detection and process the image onto the webcam stream. Only maximum of two hands can be detected
        # This is so that the Pi Theremin cannot be interfered by other hands.
#----------------------------------------------------------------
# Draw hands onto image (webcam), by converting RGB abck to BGR, as this is OpenCV's colour language and then draw the landmarks by detecting each of them using a for loop.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR) # Converts back to BGR to displayed on the opencv video frame. BGR is openCV's colour system.
        leftHand, rightHand = None, None # No detected hands on frame, initial hand values are set to None. 
        if hand_result.multi_hand_landmarks: # If hand landmarks are detected on the image:
            for landmark_item in hand_result.multi_hand_landmarks: # Loop through each landmark in then landmark list: multi_hand_landmarks.
                mp_drawing.draw_landmarks(frame, landmark_item, connections=mp_hands.HAND_CONNECTIONS) # Draw the landmarks onto the image/frame. (Optional, only used to see how mediapipe works in response to the Theremin Program.)
            leftHand = hand_result.multi_hand_landmarks[0] # Maximum number of hands used is 2,(mp.hands see above) so within the list the left hand is index 0.
            if len(hand_result.multi_hand_landmarks) == 2: # If two hands are detected within the frame,
                rightHand = hand_result.multi_hand_landmarks[1] # Identify the other hand as the right hand at index 1, as that is the second value in the list.
            
            else: # "Kinda" fixes the index out of range problem for only detecting one hand. We check how many hands are on the camera and continue the webcam stream regardless if it indexes out of range.
                continue

            # This section of code gets the x and y axis for each landmark within the frame.
            leftHandCoord = np.array([[landmark.x, landmark.y] for landmark in leftHand.landmark])
            rightHandCoord = np.array([[landmark.x, landmark.y] for landmark in rightHand.landmark])

            # minCoord and maxCoord is the result of calcualting the minimum and amximum co-ordinates of the left hand and right hand.
            # np.vstack is used to stack the co-ordinates into a 2d array in sequence vertically so it can be interpolated (next block of code).
            minCoord = np.min(np.vstack([leftHandCoord, rightHandCoord]), axis=0)
            maxCoord = np.max(np.vstack([leftHandCoord, rightHandCoord]), axis=0)

            # Map distance of hands within frame, to freq and amp ranges for the Pi Theremin. 
            # Numpy interpolation so that even if the hand coordinations fall out of range (although not expected to due to vstack and the assignment of the coordinate arrays), that it is extrapolated back into the ranges each endpoint. e.g 100,1000.
            leftHandFreq = np.interp(leftHandCoord, [minCoord[0], maxCoord[0]], [100, 1000]) # Mapping the left hand co-ordinates within the range of set amp min, and amp max 0 and 1.
            rightHandAmp = np.interp(rightHandCoord, [minCoord[1], maxCoord[1]], [0, 1])[:, np.newaxis] # np.newaxis adds an empty dimension so that numpy can interpolate rightHandAmps values to map it to the required range for amplitude of the wave.


            leftHandFreqSmoothed = np.convolve(leftHandFreq.flatten(), window, mode='same') # We flatten the left hand frequency to a 1D array so that it can be sequenced and convolved
            # Hann window used for signal processing and the same means that the input (leftHandFreq) signal will still match the output signal. This is passed as a parameter to genSineWave instead of the original left hand frequency.

            # Generate a waveform using numpy, then send it to defined pygame channel above, this is so that the sound is constant throughout hand movement,
            genSineWave(leftHandFreqSmoothed, rightHandAmp.flatten()) # We flatten the rightHandAmp parameter as the original variable takes two dimensions and must be converted to one.
            
            print(f"Left Hand Frequency: {leftHandFreq} Right Hand Frequency: {rightHandAmp}") #
        else:
            MAIN_CHANNEL.stop() # If no hands are detected, then pygames mixer will not play any sound, as there are no co-ordinates present to generate a sine wave.
        frame = cv2.flip(frame, 1) # Makes the frame horizintal, used to properly detect left and right hands.
        cv2.imshow("Pi Theremin", frame) # Display the frame, with given title "Pi Theremin."
        cv2.moveWindow("Pi Theremin", 0, 0) # Maps the frame to the top left of the window. 
        if cv2.waitKey(1) == ord('q'): # Quit program and end while loop when q is pressed.
            break

