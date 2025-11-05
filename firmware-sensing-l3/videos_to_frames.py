#!/usr/bin/env python3
"""
Extract all frames from .mp4 videos in a specified folder.
Each frame is saved as a separate PNG file in the data/frames folder.
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, video_name=None):
    """
    Extract all frames from a video file and save as PNG images.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        video_name: Optional name prefix for output files (defaults to video filename)
    
    Returns:
        int: Number of frames extracted
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if video_name is None:
        video_name = video_path.stem
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Video: {video_path.name}")
    print(f"    Resolution: {width}x{height}")
    print(f"    FPS: {fps:.2f}")
    print(f"    Total frames: {total_frames}")
    
    frame_count = 0
    
    # Extract frames with progress bar
    with tqdm(total=total_frames, desc=f"  Extracting", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save frame as PNG
            frame_filename = output_dir / f"{video_name}_frame_{frame_count:06d}.png"
            cv2.imwrite(str(frame_filename), frame)
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"    Extracted {frame_count} frames")
    
    return frame_count


def process_folder(input_folder, output_base_dir="data/frames", preserve_structure=True):
    """
    Process all .mp4 videos in a folder and extract frames.
    
    Args:
        input_folder: Folder containing .mp4 videos
        output_base_dir: Base directory for output frames
        preserve_structure: If True, maintain folder structure (e.g., data/groundhog/video.mp4 -> data/frames/groundhog/)
    """
    input_folder = Path(input_folder)
    output_base_dir = Path(output_base_dir)
    
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
    
    # Find all .mp4 files
    video_files = list(input_folder.rglob("*.mp4"))
    
    if not video_files:
        print(f"No .mp4 files found in '{input_folder}'")
        return
    
    print(f"Found {len(video_files)} video file(s) in '{input_folder}'")
    print("-" * 60)
    
    total_frames = 0
    
    for video_path in video_files:
        if preserve_structure:
            # Maintain relative folder structure
            relative_path = video_path.relative_to(input_folder)
            output_dir = output_base_dir / relative_path.parent / video_path.stem
        else:
            # Put all frames in a single folder
            output_dir = output_base_dir / video_path.stem
        
        frames_extracted = extract_frames_from_video(video_path, output_dir)
        total_frames += frames_extracted
        print()
    
    print("-" * 60)
    print(f"Total frames extracted: {total_frames}")
    print(f"Frames saved to: {output_base_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract all frames from .mp4 videos as PNG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames from all videos in data/groundhog/
  python videos_to_frames.py data/groundhog
  
  # Extract frames maintaining folder structure
  python videos_to_frames.py data --output data/frames
  
  # Extract frames from all videos in a folder and subfolders
  python videos_to_frames.py data --flat
        """
    )
    
    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder containing .mp4 videos (searches recursively)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/frames",
        help="Output directory for extracted frames (default: data/frames)"
    )
    
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Put all frames in a single folder (don't preserve structure)"
    )
    
    args = parser.parse_args()
    
    # Process videos
    process_folder(
        input_folder=args.input_folder,
        output_base_dir=args.output,
        preserve_structure=not args.flat
    )


if __name__ == "__main__":
    main()

