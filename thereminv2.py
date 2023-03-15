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

# Initialize mediapipe hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define and generate a sine wave through a function, using numpy.
def genSineWave(Freq, Amp):
    Freq = np.reshape(Freq, (-1, 1))
    SAMPLES = np.zeros((44100, 2), dtype=np.int16) # Initialize samples using a numpy array passing two channels as a tuple.
    SINE = np.sin(2 * np.pi * np.arange(44100) * Freq[0]/44100)
    SCALED_WAVE = (SINE * 32767 * Amp[1]).astype(np.int16)
    SAMPLES[:, 0] = SCALED_WAVE  # Index the samples array by 2 so that it matches the format of the steromixer.
    SAMPLES[:, 1] = SCALED_WAVE  # It duplicates the number of samples across the stereo channel 
    SOUND = pygame.sndarray.make_sound(SAMPLES) # Convert then generate the sine wave based of it's calculation of the position of the left/right hand co-ordinates.
    MAIN_CHANNEL.play(SOUND, loops = -1) # Plays the through the pygame channel.

frame = cv2.VideoCapture(0) # Open up a frame, for webcam streaming.

while True: # Loop used to run infinitely. 
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, frame = camera.read()

    # Like the old Thermein version, we draw hand landmarks onto the image by converting BGR to RGB and defining the number of hands for mediapipe to detect.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        hand_result = mp_hands.Hands(max_num_hands=2).process(frame)
#----------------------------------------------------------------
# Draw hands onto image (webcam), by converting RGB abck to BGR, as this is OpenCV's colour language and then draw the landmarks by detecting each of them using a for loop.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        leftHand, rightHand = None, None
        if hand_result.multi_hand_landmarks:
            for landmark_item in hand_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmark_item, connections=mp_hands.HAND_CONNECTIONS)
            leftHand = hand_result.multi_hand_landmarks[0]
            if len(hand_result.multi_hand_landmarks) == 2:
                rightHand = hand_result.multi_hand_landmarks[1]
            
            else: # "Kinda" fixes the index out of range problem for only detecting one hand. We check how many hands are on the camera and continue the webcam stream regardless if it indexes out of range.
                continue

            # This section of code gets the x and y axis for each landmark within the frame.
            leftHandCoord = np.array([[landmark.x, landmark.y] for landmark in leftHand.landmark])
            rightHandCoord = np.array([[landmark.x, landmark.y] for landmark in rightHand.landmark])

            minCoord = np.min(np.vstack([leftHandCoord, rightHandCoord]), axis=0)
            maxCoord = np.max(np.vstack([leftHandCoord, rightHandCoord]), axis=0)

            # Map distance of hands within frame, to freq and amp ranges for the Pi Theremin.
            leftHandFreq = np.interp(leftHandCoord, [minCoord[0], maxCoord[0]], [100, 1000])
            rightHandAmp = np.interp(rightHandCoord, [minCoord[1], maxCoord[1]], [0, 1])[:, np.newaxis]

            # Generate a waveform using numpy, then send it to defined pygame channel above, this is so that the sound is constant throughout hand movement,
            genSineWave(leftHandFreq, rightHandAmp.flatten())

            print(f"Left Hand Frequency: {leftHandFreq} Right Hand Frequency: {rightHandAmp}")
        else:
            MAIN_CHANNEL.stop()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Pi Theremin", frame)
        cv2.moveWindow("Pi Theremin", 0, 0)

        if cv2.waitKey(1) == ord('q'):
            break

