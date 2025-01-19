# TeideSat Satellite Tracking for the Optical Ground Station

This program is part of the TeideSat project, developed by a student's association from the Canary Islands' universities. The main objective of the project is to design and build a nano-satellite to be launched into space to perform scientific experiments.

This program is used to track the satellite using an Optical Ground Station (OGS), composed by a main telescope and a guider telescope. The guider telescope has a camera attached to it which is continuously recording the night sky. This video stream is feed from the OGS to this program to detect all the light sources on each frame, characterize them and obtain which of them are from moving satellites to finally track them with the main telescope.

## Installation and usage:
   1. Clone this repo:
      ```
      $ git clone https://github.com/Teidesat/teidesat-cubesat-tracker.git
      $ cd teidesat-cubesat-tracker
      ```

   2. Build the docker image:
      ```
      $ docker build -f Dockerfile -t teidesat-cubesat-tracker:latest .
      ```
      
   3. Run the docker container:
      ```
      $ docker compose run --rm teidesat-cubesat-tracker
      ```
      
## Development installation:

   1. Clone this repo:
      ```
      $ git clone https://github.com/Teidesat/teidesat-cubesat-tracker.git
      $ cd teidesat-cubesat-tracker
      ```
      
   2. Create a virtual environment:
      ```
      $ python3 -m venv venv
      $ source venv/bin/activate
      ```    

   3. Install the dependencies:
      ```
      $ pip install -r requirements.txt
      ```
      
   4. Run the code:
      ```
      $ python3 ./main.py
      ```
      or
      ```
      $ chmod +x ./main.py
      $ ./main.py
      ```

### Run the code tests:
   ```
   $ python3 -m unittest discover -s test
   ```
   or
   ```
   $ coverage run --branch --omit=config*,*init*,test* -m unittest && echo '' && coverage report -m && coverage erase
   ```

### Stars detection with the neural network model:

The neural network model used to detect stars in the video stream is a pre-trained model based on image segmentation. More information on the training process and the model architecture can be found at its dedicated GitHub repository: https://github.com/Teidesat/intelligent-sky-objects-detector

To use this detection mode, follow these steps:

   1. Download the trained neural network model from the following link:
      ```
      https://drive.google.com/drive/folders/1kO81C8dctVbbpROEhiE7BLP7iTWPYEi1?usp=sharing
      ```

   2. Create a folder named 'trained-models' and move the downloaded model to it:
      ```
      $ mkdir /path/to/trained-models
      $ mv path/to/downloaded/model /path/to/trained-models/
      ```
      
   3. Create a '.env' file with the following content:
      ```
      TRAINED_MODELS_PATH=/path/to/trained-models
      ```

   4. Verify that the detection mode is set to 'NEURAL_NETWORK' in the 'constants.py' file:
      ```
      STAR_DETECTION_MODE = "NEURAL_NETWORK"
      ```
