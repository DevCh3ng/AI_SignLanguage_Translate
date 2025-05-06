import cv2
import torch
import numpy as np
from pytorch_i3d import InceptionI3d
import os
import re
import time
import requests
import json
from requests.exceptions import RequestException
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
def generate_sentence(words):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Sign Language Sentence Builder"
            },
            data=json.dumps({
                "model": "deepseek/deepseek-r1-zero:free",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Create one complete English sentence using all of these words exactly once: {words}. Ensure the sentence is grammatically correct and makes sense. You could add words that makes the sentence grammitcally correct"
                    }
                ]
            }),
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    
    except RequestException as e:
        return f"Error generating sentence: {str(e)}"

def load_model(weights_path, num_classes=100):
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(num_classes)
    state_dict = torch.load(weights_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().cuda()
    return model

def predict_sign_language(frames, model, label_map, min_frames=30):
    while len(frames) < min_frames:
        frames.append(frames[-1])

    frames_tensor = torch.tensor(np.array(frames, dtype=np.float32)).permute(3, 0, 1, 2).unsqueeze(0).cuda()
    
    with torch.no_grad():
        logits = model(frames_tensor)
        if len(logits.shape) == 3:
            predictions = torch.mean(logits, dim=2)
        else:
            predictions = logits
        
        probs = torch.softmax(predictions, dim=1)[0]
        top_pred_idx = torch.argmax(probs).item()
        top_pred_prob = probs[top_pred_idx].item()
    
    predicted_word = re.sub(r'\d+', '', label_map[top_pred_idx]).strip()
    if predicted_word.lower() == "now":
        predicted_word = "No movement detected"
    
    return predicted_word, top_pred_prob

def prediction_stream(model, label_map, target_framerate=15, record_duration=2.0, delay_duration=1.0, display_size=(640, 480)):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, target_framerate)
    actual_framerate = cap.get(cv2.CAP_PROP_FPS)
    print(f"Set framerate: {actual_framerate} FPS (target was {target_framerate} FPS)")

    cv2.namedWindow('Sign Language Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Language Stream', display_size[0], display_size[1])

    current_prediction = "Waiting..."
    current_prob = 0.0
    saved_words = []

    print("Streaming predictions from webcam.")
    print("Press 's' to save the current prediction, 'q' to quit and generate sentence.")

    while cap.isOpened():
        sequence = []

        print("Recording for 2 seconds...")
        start_record = time.time()
        while time.time() - start_record < record_duration:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            frame_resized = cv2.resize(frame, (224, 224))
            frame_normalized = (frame_resized / 255.0) * 2 - 1
            sequence.append(frame_normalized)

            display_frame = cv2.resize(display_frame, display_size)
            cv2.rectangle(display_frame, (0, 0), (display_size[0], 100), (245, 117, 16), -1)
            cv2.putText(display_frame, "Recording...", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Saved: {', '.join(saved_words)}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Sign Language Stream', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if saved_words:
                    print("\nGenerating sentence from saved words:", saved_words)
                    sentence = generate_sentence(saved_words)
                    print("Generated sentence:", sentence)
                else:
                    print("\nNo words saved.")
                cap.release()
                cv2.destroyAllWindows()
                return

        if len(sequence) > 0:
            print(f"Captured {len(sequence)} frames. Predicting...")
            predicted_word, prob = predict_sign_language(sequence, model, label_map)
            current_prediction = predicted_word
            current_prob = prob
            print(f"Prediction: {current_prediction} | Confidence: {current_prob:.2f}")

        print("Delaying for 1 second...")
        start_delay = time.time()
        while time.time() - start_delay < delay_duration:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            frame_resized = cv2.resize(frame, (224, 224))

            display_frame = cv2.resize(display_frame, display_size)
            cv2.rectangle(display_frame, (0, 0), (display_size[0], 100), (245, 117, 16), -1)
            cv2.putText(display_frame, f"Pred: {current_prediction} ({current_prob:.2f})", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Saved: {', '.join(saved_words)}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Sign Language Stream', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and current_prediction != "Waiting..." and current_prediction != "No movement detected":
                if current_prediction not in saved_words:
                    saved_words.append(current_prediction)
                    print(f"Saved word: {current_prediction}")
            elif key == ord('q'):
                if saved_words:
                    print("\nGenerating sentence from saved words:", saved_words)
                    sentence = generate_sentence(saved_words)
                    print("Generated sentence:", sentence)
                else:
                    print("\nNo words saved.")
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    weights_path = "./weights/NSLT_2000_0.74258.pt"
    label_map_path = "class_list.txt"

    if not os.path.exists(label_map_path):
        print(f"Error: Label file not found at {label_map_path}")
        exit()
    with open(label_map_path, "r") as f:
        label_map = {i: line.strip() for i, line in enumerate(f.readlines())}

    if not os.path.exists(weights_path):
        print(f"Error: Model weights file not found at {weights_path}")
        exit()
    model = load_model(weights_path, num_classes=len(label_map))
    prediction_stream(model, label_map, target_framerate=15, record_duration=2.0, delay_duration=1.0, display_size=(1280, 720))