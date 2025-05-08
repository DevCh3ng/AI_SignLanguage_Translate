import os
import cv2
import subprocess

SUPPORTED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
def preprocess_video(input_path, output_path, target_size):
    if not os.path.isfile(input_path):
        print(f"    Skipping: Input video not found at {input_path}")
        return False

    cap_check = cv2.VideoCapture(input_path)
    if not cap_check.isOpened():
         print(f"    Error: Could not open video {input_path} with OpenCV for validation.")
         return False
    original_fps = cap_check.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_check.release()

    if total_frames <= 0:
        print(f"    Skipping: Video has 0 frames {input_path}")
        return False

    if original_fps <= 0:
        print(f"    Warning: Invalid FPS read for {input_path}. FFmpeg might guess.")
        fps_options = []
    else:
        fps_options = ['-r', str(original_fps)]

    target_width, target_height = target_size

    ffmpeg_command = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f'scale={target_width}:{target_height}',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-an',
        '-y',
        '-loglevel', 'error',
        *fps_options,
        output_path
    ]

    result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

    print(f"    Successfully processed -> {output_path}")
    return True


def process_directory(source_dir, output_dir, target_size):
    print(f"Starting preprocessing...")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target size: {target_size}")
    print("-" * 30)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        print(f"\nProcessing folder: {relative_path if relative_path != '.' else '(root)'}")

        current_output_dir = os.path.join(output_dir, relative_path)
        os.makedirs(current_output_dir, exist_ok=True)

        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in SUPPORTED_EXTENSIONS:
                input_video_path = os.path.join(root, filename)
                output_filename = os.path.splitext(filename)[0] + ".mp4"
                output_video_path = os.path.join(current_output_dir, output_filename)

                print(f"  Processing video: {filename}...")

                success = preprocess_video(input_video_path, output_video_path, target_size)
                if success:
                    processed_count += 1
                else:
                    error_count += 1
            else:
                 skipped_count +=1

    print("\n" + "-" * 30)
    print("Preprocessing finished.")
    print(f"Successfully processed: {processed_count} videos")
    print(f"Errors encountered:     {error_count} videos")
    print(f"Files skipped:          {skipped_count} (includes non-videos)")
    print("-" * 30)

if __name__ == "__main__":
    root = ""
    out = "data"
    w = 224
    h = 224

    target_wh_size = (w, h)

    if os.path.isdir(root):
        process_directory(root, out, target_wh_size)
    elif not os.path.isdir(root):
         print(f"Error: Source directory '{root}' not found or is not a directory.")
         exit()

