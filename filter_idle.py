"""
Filter idle frames from training data.
Reads h5 files and creates a new dataset with only non-idle frames.
"""
import h5py
import numpy as np
from pathlib import Path
import argparse

def filter_idle_frames(input_h5, output_h5, idle_action=0):
    """
    Filter out idle frames from h5 file.
    
    Args:
        input_h5: path to input h5 file
        output_h5: path to output h5 file
        idle_action: action class to filter out (default: 0 = idle)
    """
    print(f"Processing {input_h5}...")
    
    with h5py.File(input_h5, 'r') as f_in:
        # Get all data
        frames = f_in['frames'][:]
        actions = f_in['actions'][:]
        goals = f_in['goals'][:] if 'goals' in f_in else None
        
        # Find non-idle frames
        non_idle_mask = actions != idle_action
        non_idle_count = np.sum(non_idle_mask)
        
        print(f"  Total frames: {len(frames)}")
        print(f"  Idle frames: {np.sum(actions == idle_action)}")
        print(f"  Non-idle frames: {non_idle_count}")
        print(f"  Ratio: {non_idle_count/len(frames)*100:.1f}%")
        
        if non_idle_count == 0:
            print(f"  ⚠️  No non-idle frames! Skipping...")
            return
        
        # Create output file
        with h5py.File(output_h5, 'w') as f_out:
            f_out.create_dataset('frames', data=frames[non_idle_mask])
            f_out.create_dataset('actions', data=actions[non_idle_mask])
            if goals is not None:
                f_out.create_dataset('goals', data=goals[non_idle_mask])
            
            # Copy attributes
            for key, value in f_in.attrs.items():
                f_out.attrs[key] = value
        
        print(f"  ✅ Saved to {output_h5}")

def process_directory(input_dir, output_dir):
    """Process all h5 files in directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    h5_files = list(input_path.glob('*.h5'))
    print(f"Found {len(h5_files)} h5 files")
    
    for h5_file in h5_files:
        output_file = output_path / h5_file.name
        filter_idle_frames(h5_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='D:\\projects\\wukong_ai\\data')
    parser.add_argument('--output-dir', type=str, default='D:\\projects\\wukong_ai\\data_noidle')
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)
    print("\n✅ Done! Filtered data saved to:", args.output_dir)
