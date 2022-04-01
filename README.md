# TeideSat Satelite Tracking for the Optical Ground Station

This program implements an api to be able to follow the TeideSat's CubeSat from any Earth's location.

This is made posible by providing the ephemeris of the satelite (the elevation angle, measured from the local horizontal, and the azimuth angle, measured clockwise from the north) obtained *in real time* from the video*camera* frames *using some computer vision algorithms*.

## Installation:
1. Clone this repo:
   ```
   $ git clone https://github.com/Teidesat/teidesat-cubesat-tracker.git
   ```

2. Install depencencies:
   ```
   $ cd teidesat-cubesat-tracker
   $ pip install -r requirements.txt
   ```

## Usage:
```
$ python3 ./main.py
```
or
```
$ chmod +x ./main.py
$ ./main.py
```
