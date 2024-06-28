from deepface import DeepFace


# Function to extract face embeddings
def get_face_embeddings(frame):
    try:
        # Analyze frame using DeepFace
        result = DeepFace.represent(
            frame, enforce_detection=False)
        return result
    except Exception as e:
        print(f"Error in face embedding: {e}")
        return None

# Function to compare two faces using DeepFace


def is_same_person(face1, face2):
    try:
        result = DeepFace.verify(
            face1, face2, enforce_detection=False)
        return result['verified']
    except Exception as e:
        raise Exception("exception during is_same_person: ", e)
        # print(f"Error in face verification: {e}")
        return False
