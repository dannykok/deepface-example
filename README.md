# Face Detection and Recognition Project

This is an example project that discover and return any unique faces in a given video, using (deepface)[https://github.com/serengil/deepface].

## Features

- **Face Detection**: Locate faces in any given image.
- **Face Recognition**: Verify the identity of individuals by comparing faces against known identities.
- **Anti-Spoofing**: Optional anti-spoofing feature to enhance security by preventing fake identity verification.

## Getting Started

### Prerequisites

- Python 3.x
- Necessary Python packages as listed in `requirements.txt`. Install them using the command:
  ````sh
  pip install -r requirements.txt
    ```
  ````

## Installation

```
git clone <repository_url>
pip install -r requirements.txt
```

## Usage

To run the face detection and recognition:

```
python main.py path/to/video
```

## Configuration

The project allows for various configurations, including the choice of face recognition model and similarity metric. These can be adjusted in the `main.py` script.

## Output

Images of unique faces found will be output into the `output_faces` directory.

## Contributing

Contributions to improve the project are welcome. Please follow the standard fork-and-pull request workflow.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. However, it's important to note that this project depends on the DeepFace library. You should review its license and terms of use before using this project.

## Acknowledgments

Thanks to the open-source community for providing the tools and libraries used in this project.
