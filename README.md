# TRY THY EYE

## Virtual Glasses Try-On

Ever wondered how a pair of glasses would look on you—or your friend—before buying them?  
This project brings that idea to life! Using real-time face tracking and sunglasses segmentation, you can virtually “try on” glasses and see exactly how they fit, rotate, and move with your face.

---
## Dataset Used
- https://universe.roboflow.com/doms/eyeglass-9xul2

---
  
## Features

- Real-time webcam support: See the glasses overlay live as you move.  
- Smart placement: Glasses stick to your eyes and rotate naturally as you tilt your head.  
- Size adjustment: Increase or decrease the size of the glasses dynamically.  
- Transparent backgrounds: Glasses are cut out cleanly from the original images, so they overlay naturally.  
- Multiple faces: Works for more than one face at a time.  

---

## How It Works

- Sunglasses segmentation: YOLOv8 with a custom segmentation model isolates the glasses from any image you provide.  
- Face tracking: Mediapipe’s Face Mesh detects key facial landmarks (eyes, nose).  
- Smart overlay:  
  - Glasses are resized according to the distance between eyes.  
  - They rotate to match your head tilt.  
  - They follow your face as you move.  
  - Alpha blending ensures the glasses appear naturally without harsh edges.  

---

## Getting Started

### Requirements

- Python 3.10+  
- OpenCV  
- Mediapipe  
- Ultralytics YOLO  
- NumPy  

---

## Usage

1. Clone the repository and place your images in the `images_test/` folder.  
2. Train or load a YOLOv8 segmentation model for sunglasses using the provided scripts.  
3. Run the main webcam script to see glasses overlay in real time.  

---

## Controls

- M – Start the glasses overlay  
- N – Stop the overlay  
- I – Increase glasses size  
- D – Decrease glasses size  
- Q – Quit  

---

## Example

Load an image of a friend’s sunglasses and see it on your face in real time using your webcam.

---

## Why I Made This

Before buying glasses, we often borrow from friends or try them in stores.  
I wanted to replicate that digital try-on experience—see how glasses suit your face, test rotation, and adjust size—all from your webcam.  

---

## Future Improvements

- Support for multiple glasses styles at once  
- Better edge smoothing to avoid skin bleed-through  
- Mobile/web app integration for instant try-on  

---

## License

This project is open source. Feel free to use and modify it for personal or educational purposes.











