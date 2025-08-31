# Video-Based Molecular Visualizer

This directory contains the video-based molecular visualization system that replaces the OpenGL 3D rendering approach.

## Structure

- `video_molecular_viewer.py` - Main application with video playback
- `videos/` - Directory for molecular animation videos
- `video_generator.py` - Helper script to create placeholder videos or download them

## Video Requirements

Videos should be:
- Format: MP4 (H.264)
- Resolution: 1920x1080 or 1280x720
- Duration: 10-30 seconds (will loop automatically)
- Naming: `{molecule}_rotation.mp4` (e.g., `h2_rotation.mp4`)

## Supported Molecules

- H2 (Hydrogen gas)
- NH3 (Ammonia)
- CH4 (Methane)
- H2O (Water)
- CO2 (Carbon dioxide)
- C2H6 (Ethane)

## Usage

1. Place molecular rotation videos in the `videos/` directory
2. Run: `python gui/video_molecular_viewer.py`
3. Use the dropdown to select different molecules
4. Videos will loop continuously as background

## Creating Videos

You can create molecular videos using:
- Blender with molecular modeling add-ons
- ChemSketch or similar chemistry software
- Online molecular viewers with screen recording
- AI-generated molecular animations

The video generator script can help create placeholder content if needed.
