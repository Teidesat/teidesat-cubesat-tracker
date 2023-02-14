# TeideSat Satellite Tracking for the Optical Ground Station

The TeideSat CubeSat tracking program offers an API that enables real-time monitoring from any location on Earth.

Utilizing computer vision algorithms, this program provides the satellite’s ephemeris, which includes the elevation angle and azimuth angle. These values, obtained in real-time from video camera frames and TLE coordinates, allow for accurate and up-to-date tracking of the TeideSat CubeSat.

## Installation:
1. Clone this repo:
   ```
   $ git clone https://github.com/Teidesat/teidesat-cubesat-tracker.git
   ```

2. Install dependencies:
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
