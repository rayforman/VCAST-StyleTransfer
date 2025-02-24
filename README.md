# V-CAST: Video Style Transfer Application

V-CAST is an extension of the CAST (Contrastive Arbitrary Style Transfer) framework that enables real-time video style transfer. This application allows users to transform their videos using various artistic styles, similar to how popular image filters work, but for video content.

## 💡 Inspiration

<img src="/public/spiderverse.png" alt="Spiderverse" width="40%"/>
<img src="/public/arts.png" alt="Arts" width="35%"/>

This project was inspired by the many art styles present in Sony Pictures’ Spider-Man: Across the Spider-Verse (2023) which pushed the boundary of what is possible in stylized animation. With a proper video style transfer network more films could more easily incorporate mixed styles in their work. 


## 🎨 Features

- Upload and process MP4 video files
- Apply artistic style transfers from reference images
- Real-time video processing
- User-friendly GUI interface built with TKinter
- Optical flow-based frame consistency
- Support for various artistic styles

## 🛠 Technologies

- **Python** - Core programming language
- **TKinter** - GUI framework
- **PyAV** - Video processing library
- **CAST Framework** - Base style transfer implementation
- **Optical Flow** - Frame consistency algorithm

## 📋 Prerequisites

- Python 3.7+
- PyAV library
- TKinter (usually comes with Python)
- Required Python packages (see requirements.txt)

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/v-cast.git
cd v-cast
```

2. Install PyAV:
```bash
pip install av
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the application:
```bash
python app.py
```

2. Use the GUI to:
   - Select an MP4 file using the "Upload Video" button
   - Choose a style image using the "Upload Style Image" button
   - Click "Stylize" to process the video

3. Find your stylized video:
   - The output will be saved as 'out_video.mp4' in the 'VCAST-APP' folder

### Application Interface
![V-CAST Demo](/public/vcast-demo.mp4)

## 📁 Project Structure

```
v-cast/
├── app.py              # Main application file
├── gifs/               # Example input videos
├── images/             # Example style images
├── models/             # Neural network models
└── utils/              # Helper functions
```

## 🎯 Use Cases

- Converting regular videos into animated style
- Applying artistic filters to video content
- Creating stylized video content for social media
- Educational demonstrations of style transfer
- Artistic video projects

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original CAST framework developers
- PyAV development team
- Contributors and testers

## 👥 Team

Built with love by Stephen Pasch, Alessandro Castillo, Raymond Forman

Columbia University Computer Science Department