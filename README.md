# VirtualFittingRoom
### Real-Time 3D Human Tracking and Virtual Try-On System

VirtualFittingRoom is an experimental real-time 3D virtual fitting room project that aims to bridge computer vision, human pose estimation, 3D character animation, and virtual garment visualization.

The project uses a camera to capture a person's movements, processes the video through computer vision and pose-estimation techniques, and transfers the detected body movements to a 3D human avatar inside Blender. The long-term objective is to develop a system in which a digital avatar can represent a user's body proportions, movements, and appearance, allowing users to visualize how different 3D garments may fit their bodies.

The current implementation focuses on webcam-based human pose detection, real-time communication between Python and Blender, and movement retargeting onto an existing 3D human rig.

---

## 1. Project Objectives

The primary objective is to develop an accessible, real-time virtual fitting room that does not depend entirely on expensive 3D scanning equipment or manually created avatars.

The system is being developed around the following objectives:

- Capture human body movements using a regular camera or webcam.
- Detect important human body landmarks using computer vision.
- Process pose information in real time.
- Transfer detected movements to an existing 3D human character.
- Develop a personalized 3D avatar that can adapt to an individual's body proportions.
- Investigate realistic garment fitting and visualization using 3D clothing models.
- Build a foundation that can eventually support interactive virtual try-on applications.

## 2. Current Features

The project currently includes the following components:

- **Camera Input:** Captures live video from a connected camera.
- **Human Pose Detection:** Uses MediaPipe to identify human body landmarks.
- **Landmark Processing:** Processes detected body coordinates and confidence values.
- **3D Pose Communication:** Transfers tracking information from Python to Blender through a local TCP connection.
- **Blender Integration:** Receives live tracking information inside Blender.
- **3D Avatar Integration:** Uses a MakeHuman-compatible human mesh and armature.
- **Live Arm Retargeting:** Applies detected arm movement to the existing 3D character using inverse kinematics.
- **Head Movement:** Includes an initial head-orientation estimation based on facial landmark positions.
- **Camera Dashboard:** Provides an interface for monitoring the camera-tracking process.

**Development status:** The camera tracking and Python-to-Blender communication pipeline are implemented. Arm and head retargeting are under development and require further calibration for accurate movement reproduction.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main application and tracking logic |
| OpenCV | Camera input and image processing |
| MediaPipe | Human pose landmark detection |
| Blender 5.2.2 | 3D character environment and animation |
| MakeHuman | Human avatar and rig generation |
| NumPy | Numerical and coordinate processing |
| TCP Socket Communication | Real-time communication between Python and Blender |
| Linux / Ubuntu | Current development environment |
| Git and GitHub | Version control and project distribution |

Additional libraries may be required depending on the selected model and future development stages.

## 4. System Architecture

The project follows a modular architecture in which camera processing, pose estimation, data transmission, and 3D animation operate as separate components.

```text
                 REAL-WORLD USER
                        |
                        v
                  WEBCAM / CAMERA
                        |
                        v
                  OPENCV CAPTURE
                        |
                        v
               MEDIAPIPE POSE DETECTION
                        |
                        v
                LANDMARK PROCESSING
                        |
                        v
               PYTHON TRACKING SERVICE
                        |
                        v
                TCP SOCKET CONNECTION
                  127.0.0.1:8766
                        |
                        v
                BLENDER POSE RECEIVER
                        |
                        v
                MOVEMENT RETARGETING
                        |
                        v
                 INVERSE KINEMATICS
                        |
                        v
                  3D HUMAN AVATAR
                        |
                        v
                 LIVE ANIMATION
```

The architecture is modular so that individual components can be improved or replaced without redesigning the entire application.

## 5. Requirements

### Hardware

Recommended hardware for development:

- A computer capable of running Blender.
- A compatible webcam or external camera.
- At least 8 GB RAM; 16 GB or more is recommended.
- A dedicated GPU is beneficial for more demanding 3D and machine-learning workloads.
- A camera position that allows the required body landmarks to remain visible.

A high-frame-rate camera may improve motion capture responsiveness, but 120 FPS is not a mandatory requirement for the current basic pose-tracking implementation.

### Software

The current development setup uses:

- Ubuntu Linux.
- Python 3.
- Blender 5.2.2.
- Git.
- OpenCV.
- MediaPipe.

The installation instructions below target Ubuntu/Linux. Other operating systems have not yet been fully validated.

## 6. Installation Guide

### Step 1: Install system dependencies

Open a terminal and execute:

```bash
sudo apt update

sudo apt install -y \
    git \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    v4l-utils
```

### Step 2: Clone the repository

Replace `YOUR_USERNAME` with the GitHub username or organization that owns the repository.

```bash
git clone https://github.com/YOUR_USERNAME/VirtualFittingRoom.git

cd VirtualFittingRoom
```

### Step 3: Create a Python virtual environment

```bash
python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip
```

### Step 4: Install Python dependencies

Install the dependencies listed in the repository:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain the tested and compatible package versions for the published release.

### Step 5: Install Blender

Download and install Blender 5.2.2 or another version explicitly supported by the repository.

Verify the installation:

```bash
blender --version
```

Make sure the Blender executable is accessible from the terminal or configure its location in the project's configuration file.

### Step 6: Configure the camera

Connect a webcam or compatible camera.

Check the available camera devices:

```bash
ls /dev/video*
```

To inspect supported camera formats and frame rates:

```bash
v4l2-ctl --list-formats-ext
```

Select a supported camera device and resolution for the tracking application.

### Step 7: Prepare the 3D avatar

The current Blender retargeting implementation expects a compatible MakeHuman-style avatar.

The Blender scene should contain:

- A human mesh named `Human`.
- An armature named `Human.rig`.
- Compatible armature bone names and hierarchy.
- The required arm and IK target objects.

The existing retargeting scripts depend on specific bone and object names. A different avatar or armature may require additional rig mapping and calibration.

If the repository does not include the avatar assets, follow the instructions in the `models/README.md` file to obtain and prepare them.

## 7. Running the Application

The application uses separate processes for camera tracking and Blender animation.

### Terminal 1: Start the camera tracking service

Activate the environment:

```bash
cd VirtualFittingRoom

source .venv/bin/activate
```

Start the camera dashboard using the repository's launch script:

```bash
./launch_dashboard.sh
```

If the launch script is located in another directory, use the corresponding documented path.

### Terminal 2: Start Blender

Open Blender and load the prepared VirtualFittingRoom scene.

Ensure that the Blender pose receiver and retargeting scripts are initialized.

The camera service and Blender receiver communicate through:

```text
Host: 127.0.0.1
Port: 8766
Protocol: TCP
```

Only one receiver should use the configured port at a time.

Once both processes are running, the camera should begin sending pose information to Blender, where the existing 3D character can respond to the detected movements.

## 8. Camera Positioning and Tracking Guidelines

Tracking accuracy depends heavily on camera placement and visibility.

For reliable pose detection:

1. Position the camera so that the required body parts remain inside the image.
2. Keep the shoulders, elbows, and wrists visible when testing arm movement.
3. Avoid placing the subject too close to the camera.
4. Use sufficient lighting.
5. Avoid backgrounds that make the person's body difficult to distinguish.
6. Keep the camera stable during tracking.

MediaPipe provides landmark confidence values. When a body part is outside the image or occluded, its confidence may decrease, resulting in inaccurate or incomplete avatar movement.

## 9. Project Directory Structure

The following is the recommended organization of the repository. Actual file locations should match the published implementation.

```text
VirtualFittingRoom/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env.example
│
├── docs/
│   ├── installation.md
│   ├── architecture.md
│   ├── troubleshooting.md
│   └── hardware-requirements.md
│
├── src/
│   ├── camera/
│   │   ├── dashboard.py
│   │   ├── service.py
│   │   ├── tracking.py
│   │   └── protocol.py
│   │
│   ├── blender_pose_receiver.py
│   ├── blender_arms_head.py
│   └── blender_vfr_startup.py
│
├── models/
│   └── README.md
│
├── assets/
│   └── README.md
│
├── scripts/
│   ├── setup_environment.sh
│   ├── launch_dashboard.sh
│   └── launch_blender.sh
│
└── tests/
    └── README.md
```

This is a recommended repository structure, not a guarantee that every listed file is already included.

## 10. Development Roadmap

The project is being developed incrementally.

### Stage 1: Camera and Pose Detection

- Webcam integration.
- Real-time image capture.
- Human pose landmark detection.
- Landmark confidence evaluation.
- Coordinate processing.

### Stage 2: Blender Integration

- Python-to-Blender communication.
- Live pose data reception.
- 3D skeleton visualization.
- Existing human rig integration.
- Initial arm and head movement retargeting.

### Stage 3: Improved Avatar Retargeting

- Accurate shoulder, elbow, and wrist mapping.
- Full-body movement tracking.
- Torso and leg movement.
- Joint orientation calibration.
- Improved motion smoothing.
- Better handling of occluded landmarks.

### Stage 4: Personalized 3D Avatar

- Human body proportion estimation.
- Body-shape adaptation.
- Facial feature reconstruction.
- Skin appearance estimation.
- Personalized avatar generation.

### Stage 5: Virtual Garment Fitting

- 3D garment asset integration.
- Garment-to-avatar alignment.
- Clothing deformation.
- Collision handling.
- Realistic fitting visualization.

### Stage 6: Real-Time Virtual Fitting Room

- Interactive virtual try-on interface.
- Multiple clothing options.
- Different viewing angles.
- Improved real-time performance.
- Potential web-based and mobile integration.

The later stages are development objectives and are not yet implemented as complete features.

## 11. Configuration and Portability

The project is intended to support different development environments.

Before running the application, developers may need to configure:

- Camera device index.
- Camera resolution and frame rate.
- Blender executable path.
- Project and model directories.
- TCP host and port.
- Tracking confidence thresholds.

Personal computer paths should not be hardcoded into the published source code. Configuration should be handled through relative paths and environment variables wherever possible.

Example configuration:

```env
CAMERA_INDEX=0
BLENDER_HOST=127.0.0.1
BLENDER_PORT=8766
POSE_CONFIDENCE=0.55
```

The actual configuration variables must match those supported by the implementation.

## 12. Troubleshooting

**Camera not detected**

- Check whether the camera is connected.
- Run `ls /dev/video*`.
- Verify camera permissions.
- Check whether another application is already using the camera.

**Blender cannot connect to the tracking service**

- Confirm that the camera service is running.
- Check that both processes use the same TCP port.
- Verify that port `8766` is not occupied by another process.
- Ensure that only one Blender receiver is running.

**Avatar does not move**

- Confirm that pose packets are reaching Blender.
- Check the receiver's active status.
- Verify the rig and bone names.
- Confirm that the required IK targets exist.

**Arms or wrists move incorrectly**

- Ensure that the elbows and wrists remain visible.
- Check MediaPipe landmark confidence values.
- Verify the mapping between camera coordinates and Blender coordinates.
- Check the IK target references and armature configuration.

**Performance problems**

- Reduce camera resolution if necessary.
- Close unnecessary applications.
- Check CPU and GPU utilization.
- Select a camera frame rate supported by the hardware.

For additional issues, consult the documentation or open a GitHub issue with relevant logs and system information.

## 13. Privacy

The current architecture processes camera input locally and sends pose information to the local Blender receiver.

Developers should review the camera service, network configuration, and any additional integrations before deployment. Avoid publishing personal camera recordings, facial data, or other identifiable user information without appropriate consent.

## 14. Contributing

Contributions, suggestions, bug reports, and technical improvements are welcome.

To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Make your changes.
4. Test the implementation.
5. Commit your changes with a meaningful message.
6. Submit a pull request describing the changes.

Please include relevant environment details, reproduction steps, and error logs when reporting bugs.

## 15. Licensing and Third-Party Assets

The project source code should be distributed under the license specified in the repository's `LICENSE` file.

Third-party software, pretrained models, human meshes, avatar assets, and datasets may have separate licensing and redistribution requirements. Users must review the applicable licenses before downloading, modifying, or redistributing these assets.

Do not assume that an asset can be redistributed simply because it is freely downloadable.

## 16. Project Status

VirtualFittingRoom is an active experimental development project.

The current focus is improving the reliability of camera-based human pose estimation and transferring detected movements to an existing Blender human rig.

The broader goal is to create a personalized digital human avatar that can eventually support realistic 3D garment fitting and real-time virtual try-on.

The project is open to researchers, developers, students, and contributors interested in computer vision, 3D graphics, human motion tracking, digital avatars, and virtual fitting room technologies.

---

**Repository:** https://github.com/YOUR_USERNAME/VirtualFittingRoom

**Project Name:** VirtualFittingRoom

**Primary Development Environment:** Ubuntu Linux

**Primary 3D Software:** Blender

**Primary Pose Estimation Framework:** MediaPipe

**Development Status:** Experimental / Under Active Development
