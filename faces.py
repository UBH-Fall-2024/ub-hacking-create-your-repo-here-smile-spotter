import cv2
import face_recognition
import numpy as np
from datetime import datetime
import os
import mss

frame_count = 0
scale_factor = 1

# Path to store profiles in a folder on the desktop
desktop_path = os.path.expanduser("~/Desktop")
output_folder = os.path.join(desktop_path, "detected_faces")
os.makedirs(output_folder, exist_ok=True)
profiles_folder = os.path.join(output_folder, "profiles")
os.makedirs(profiles_folder, exist_ok=True)

# Lists to store known face encodings, profile names, and quality info
known_face_encodings = []
known_face_names = []
known_face_quality = {}  # Dictionary to store profile_name -> quality score

def calculate_image_quality(face_image):
    """Calculate image quality based on sharpness or clarity."""
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()  # Sharpness metric
    return laplacian_var

def is_face_forward_facing(landmarks):
    """Check if a face is forward-facing based on facial landmarks."""
    try:
        if "left_eye" in landmarks and "right_eye" in landmarks and "nose_bridge" in landmarks:
            left_eye = np.mean(landmarks["left_eye"], axis=0)
            right_eye = np.mean(landmarks["right_eye"], axis=0)
            nose_bridge = landmarks["nose_bridge"][0]  # Use the top of the nose bridge

            # Check horizontal alignment of eyes and nose bridge position
            eye_distance = np.linalg.norm(left_eye - right_eye)
            nose_to_eyes_distance = np.abs((left_eye[1] + right_eye[1]) / 2 - nose_bridge[1])

            # Adjust thresholds based on testing; lower values enforce stricter alignment
            if nose_to_eyes_distance < eye_distance * 0.1:  # Nose roughly centered
                return True
        return False
    except Exception as e:
        print(f"Error in is_face_forward_facing: {e}")
        return False

def save_or_replace_face(face_image, profile_name):
    """Save a new face image if it's of better quality than the current one."""
    profile_folder = os.path.join(profiles_folder, profile_name)
    os.makedirs(profile_folder, exist_ok=True)

    # Calculate quality of the current face image
    current_quality = calculate_image_quality(face_image)

    # Check if this profile already has a saved image
    if profile_name in known_face_quality:
        if current_quality > known_face_quality[profile_name]:  # Replace if quality is higher
            # Replace the saved image with the new higher-quality image
            filename = f"{profile_folder}/best_face.jpg"
            cv2.imwrite(filename, face_image)
            known_face_quality[profile_name] = current_quality
            print(f"Updated {profile_name} with a higher-quality image.")
    else:
        # If no image exists, save this one as the initial image
        filename = f"{profile_folder}/best_face.jpg"
        cv2.imwrite(filename, face_image)
        known_face_quality[profile_name] = current_quality
        print(f"Saved initial image for {profile_name}")

def facializer(rgb_img, img, output_folder):
    try:
        small_img = cv2.resize(rgb_img, (0, 0), fx=scale_factor, fy=scale_factor)
        face_locations = face_recognition.face_locations(small_img)
        face_encodings = face_recognition.face_encodings(small_img, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(small_img, face_locations)

        # Scale face locations back to original size if the scale factor is not 1
        if scale_factor != 1:
            face_locations = [
                (int(top / scale_factor), int(right / scale_factor), int(bottom / scale_factor), int(left / scale_factor))
                for top, right, bottom, left in face_locations
            ]
        
        for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
            # Check if the face is forward-facing
            if not is_face_forward_facing(landmarks):
                print("Face not forward-facing. Skipping...")
                continue  # Skip saving if the face is not forward-facing

            # Use a strict tolerance to find potential matches
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # Set a threshold distance for strict face matching
            threshold_distance = 0.55
            best_match_index = None
            
            # Find the best match based on the smallest distance
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] > threshold_distance:
                    best_match_index = None  # No good match found within threshold

            if best_match_index is not None and matches[best_match_index]:
                # If a match is found, use known profile
                profile_name = known_face_names[best_match_index]
            else:
                # Create a new profile for a new face
                profile_name = f"profile_{len(known_face_names) + 1}"
                known_face_encodings.append(face_encoding)
                known_face_names.append(profile_name)
                print(f"New profile created: {profile_name}")
            
            # Extract the face image from the original frame
            face_image = img[top:bottom, left:right]
            
            # Save or replace the image in the profile folder based on quality
            save_or_replace_face(face_image, profile_name)
            
            # Draw rectangle around face and display profile name
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, profile_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    except Exception as e:
        print(f"Error in facializer: {e}")

# Start screen capture session
with mss.mss() as sct:
    monitor = {"top": 100, "left": 100, "width": 1800, "height": 1169}
    
    while True:
        # Capture screen region and convert it to a NumPy array
        img = np.array(sct.grab(monitor))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # Process every 10th frame
        if frame_count % 1 == 0:
            facializer(rgb_img, img, output_folder)
        
        # Display the image with detected faces
        #cv2.imshow("Facial Detection", img)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        # Increment frame count for frame skipping
        frame_count += 1

# Release resources and close windows
cv2.destroyAllWindows()
