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
