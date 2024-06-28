import argparse
import cv2
import os

from face_detect import get_face_embeddings, is_same_person


def scan_video(args):

    # Initialize video capture
    video_capture = cv2.VideoCapture(args.video_path)

    # Initialize variables
    face_encodings = []

    # Create a directory to save the faces
    face_output_dir = args.face_output_dir
    os.makedirs(face_output_dir, exist_ok=True)

    frame_output_dir = args.frame_output_dir
    if frame_output_dir is not None:
        os.makedirs(frame_output_dir, exist_ok=True)

    frame_count = 0
    processed_frame_count = 0
    skip_frames = args.skip_frame  # Process every 5th frame

    while True:
        ret = video_capture.grab()
        if not ret:
            break
        frame_count += 1

        if frame_count % skip_frames == 0:  # processing frame
            processed_frame_count += 1

            status, frame = video_capture.retrieve()  # Decode processing frame

            # Convert the frame to RGB (DeepFace expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect and encode faces
            embeddings = get_face_embeddings(rgb_frame)
            if embeddings is not None:
                face_found = 0
                for emb in embeddings:
                    if emb["face_confidence"] >= 0.95:
                        # save frame_num to dict
                        emb['frame_num'] = frame_count
                        face_encodings.append(emb)
                        face_found += 1
                print(
                    f"Found {face_found} faces in frame {frame_count}")

                # Draw bounding boxes, with the confidence score on the frame
                if embeddings is not None:
                    for face_data in embeddings:
                        x = face_data['facial_area']['x']
                        y = face_data['facial_area']['y']
                        w = face_data['facial_area']['w']
                        h = face_data['facial_area']['h']
                        conf = face_data['face_confidence']
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 0, 255), 2)
                        cv2.putText(frame, f"{conf:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        # save the frame
                        if frame_output_dir:
                            cv2.imwrite(
                                f'{frame_output_dir}/frame_{frame_count}.jpg', frame)

    print("Extracting unique faces...")
    # Extract embeddings and remove duplicates
    unique_face_encodings = []

    for encoding in face_encodings:
        is_unique = True
        for unique_encoding in unique_face_encodings:
            # Compare the embeddings
            # DeepFace.verify accepts List as precalulated embeddings
            face1 = encoding['embedding']
            face2 = unique_encoding['embedding']
            if is_same_person(face1, face2):
                is_unique = False
                break
        if is_unique:
            unique_face_encodings.append(encoding)

    print(f"Found {len(unique_face_encodings)} unique faces in the video.")
    # save the frame with face only
    face_id = 0
    for encoding in unique_face_encodings:
        print("looping through unique_face_encodings")
        frame_num = encoding['frame_num']
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video_capture.read()
        if not ret:
            print("Error reading video frame while doing the final capture. Skipping...")
            continue

        x = encoding['facial_area']['x']
        y = encoding['facial_area']['y']
        w = encoding['facial_area']['w']
        h = encoding['facial_area']['h']
        face = frame[y:y+h, x:x+w]
        print(f"Saving face_{face_id}.jpg")
        cv2.imwrite(f'{face_output_dir}/face_{face_id}.jpg', face)
        face_id += 1

    # Release video capture
    video_capture.release()

    # Save the frame to the output directory
    # output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    # cv2.imwrite(output_path, frame)
    # frame_count += 1

    # Save the face inside the frame


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Extract unique faces from a video')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--skip_frame', type=int, default=1,
                        help='number of frames to be skipped during sampling')
    parser.add_argument('--face_output_dir', type=str,
                        default='output_faces', help='Path to save the faces')
    parser.add_argument('--frame_output_dir', type=str, default=None,
                        help='Path to save the frames. Default is None which doesn not save frames')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()

    # calucate time taken
    import time
    start = time.time()
    scan_video(args)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
