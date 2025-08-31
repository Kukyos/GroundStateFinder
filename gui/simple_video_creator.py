#!/usr/bin/env python3
"""
Simple video creator using PIL/Pillow to create animated molecular videos
"""
from PIL import Image, ImageDraw
import os
import math

def create_simple_molecular_video(molecule_name, output_path, frames=300):
    """Create a simple animated molecular video using PIL."""
    print(f"Creating simple video for {molecule_name}...")
    
    # Video dimensions
    width, height = 1280, 720
    
    # Colors for different molecules
    colors = {
        "h2": [(255, 255, 255), (255, 255, 255)],  # White hydrogens
        "nh3": [(50, 50, 255), (255, 255, 255)],   # Blue nitrogen, white hydrogens
        "ch4": [(100, 100, 100), (255, 255, 255)], # Gray carbon, white hydrogens
        "h2o": [(255, 50, 50), (255, 255, 255)],   # Red oxygen, white hydrogens
        "co2": [(100, 100, 100), (255, 50, 50)],   # Gray carbon, red oxygens
        "c2h6": [(100, 100, 100), (255, 255, 255)] # Gray carbons, white hydrogens
    }
    
    molecule_colors = colors.get(molecule_name, [(255, 255, 255), (200, 200, 200)])
    
    frames_list = []
    
    for frame in range(frames):
        # Create image
        img = Image.new('RGB', (width, height), (12, 8, 20))  # Dark background
        draw = ImageDraw.Draw(img)
        
        # Calculate rotation
        angle = (frame / frames) * 2 * math.pi
        
        center_x, center_y = width // 2, height // 2
        
        if molecule_name == "h2":
            # Two atoms
            radius = 80
            atom1_x = center_x + radius * math.cos(angle)
            atom1_y = center_y
            atom2_x = center_x - radius * math.cos(angle)
            atom2_y = center_y
            
            # Draw bond
            draw.line([(atom1_x, atom1_y), (atom2_x, atom2_y)], fill=(255, 255, 0), width=8)
            
            # Draw atoms
            draw.ellipse([atom1_x-30, atom1_y-30, atom1_x+30, atom1_y+30], fill=molecule_colors[0])
            draw.ellipse([atom2_x-30, atom2_y-30, atom2_x+30, atom2_y+30], fill=molecule_colors[1])
            
        elif molecule_name == "nh3":
            # Nitrogen in center, 3 hydrogens around
            n_x, n_y = center_x, center_y
            radius = 100
            
            # Draw nitrogen
            draw.ellipse([n_x-40, n_y-40, n_x+40, n_y+40], fill=molecule_colors[0])
            
            # Draw hydrogens
            for i in range(3):
                h_angle = angle + (i * 2 * math.pi / 3)
                h_x = n_x + radius * math.cos(h_angle)
                h_y = n_y + radius * math.sin(h_angle)
                
                # Bond
                draw.line([(n_x, n_y), (h_x, h_y)], fill=(255, 255, 0), width=6)
                # Hydrogen
                draw.ellipse([h_x-25, h_y-25, h_x+25, h_y+25], fill=molecule_colors[1])
        
        # Add frame to list
        frames_list.append(img)
    
    # Save as animated GIF (can be converted to MP4 later)
    gif_path = output_path.replace('.mp4', '.gif')
    frames_list[0].save(
        gif_path,
        save_all=True,
        append_images=frames_list[1:],
        duration=50,  # 50ms per frame = 20fps
        loop=0
    )
    
    print(f"Created animated GIF: {gif_path}")
    print("Note: For MP4 conversion, use FFmpeg or similar tool")
    
    return gif_path

def main():
    """Create simple molecular videos."""
    videos_dir = os.path.join("gui", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    molecules = ["h2", "nh3", "ch4", "h2o", "co2"]
    
    for molecule in molecules:
        output_path = os.path.join(videos_dir, f"{molecule}_rotation.mp4")
        create_simple_molecular_video(molecule, output_path)

if __name__ == "__main__":
    main()
