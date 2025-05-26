import os
import re
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def get_tensorboard_data(tensorboard_path):
    """
    Extract data from tensorboard logs
    
    Args:
        tensorboard_path: Path to the tensorboard log directory
    
    Returns:
        Dictionary with extracted data
    """
    ea = event_accumulator.EventAccumulator(
        tensorboard_path, 
        size_guidance={event_accumulator.SCALARS: 0}  # Load all scalar events
    )
    ea.Reload()
    
    # Get all scalar tags (these contain the alignment error data)
    scalar_tags = ea.Tags()['scalars']
    
    # Filter for alignment error tags
    alignment_tags = [tag for tag in scalar_tags if 'AlignmentErrors_mrad' in tag]
    
    # Extract data from tags
    data = {}
    for tag in alignment_tags:
        events = ea.Scalars(tag)
        
        # Get the last epoch's data (highest step value)
        if events:
            last_event = events[-1]
            
            # Parse tag to extract mode, heliostat_id, and calib_id
            # Format: AlignmentErrors_mrad/{mode}/{heliostat_id}/{calib_id}
            parts = tag.split('/')
            if len(parts) >= 4:
                mode = parts[1]
                heliostat_id = parts[2]
                calib_id = parts[3]
                
                # Store data
                if heliostat_id not in data:
                    data[heliostat_id] = {}
                
                if mode not in data[heliostat_id]:
                    data[heliostat_id][mode] = {}
                
                # Store the error value at the last epoch
                data[heliostat_id][mode][calib_id] = {
                    'error': last_event.value,
                    'epoch': last_event.step
                }
    
    return data

def get_heliostat_ids(tensorboard_path):
    """
    Extract all unique heliostat IDs from tensorboard logs
    
    Args:
        tensorboard_path: Path to the tensorboard log directory
    
    Returns:
        List of heliostat IDs
    """
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    
    scalar_tags = ea.Tags()['scalars']
    
    # Extract heliostat IDs from tags
    heliostat_ids = set()
    pattern = r'AlignmentErrors_mrad/[^/]+/([^/]+)/'
    
    for tag in scalar_tags:
        match = re.search(pattern, tag)
        if match:
            heliostat_ids.add(match.group(1))
    
    return sorted(list(heliostat_ids))

def get_calib_ids(tensorboard_path, heliostat_id=None, mode=None):
    """
    Extract calibration IDs for a specific heliostat and/or mode
    
    Args:
        tensorboard_path: Path to the tensorboard log directory
        heliostat_id: Optional, filter by heliostat ID
        mode: Optional, filter by mode (Train/Valid/Test)
    
    Returns:
        Dictionary mapping modes to lists of calibration IDs,
        or a list of calibration IDs if mode is specified
    """
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    
    scalar_tags = ea.Tags()['scalars']
    
    # Build regex pattern based on filters
    if heliostat_id and mode:
        pattern = f'AlignmentErrors_mrad/{mode}/{heliostat_id}/([^/]+)'
        calib_ids = set()
        
        for tag in scalar_tags:
            match = re.search(pattern, tag)
            if match:
                calib_ids.add(match.group(1))
        
        return sorted(list(calib_ids))
    
    elif heliostat_id:
        result = {}
        mode_pattern = f'AlignmentErrors_mrad/([^/]+)/{heliostat_id}/([^/]+)'
        
        for tag in scalar_tags:
            match = re.search(mode_pattern, tag)
            if match:
                mode = match.group(1)
                calib_id = match.group(2)
                
                if mode not in result:
                    result[mode] = set()
                
                result[mode].add(calib_id)
        
        # Convert sets to sorted lists
        for mode in result:
            result[mode] = sorted(list(result[mode]))
        
        return result
    
    else:
        # Return all calib_ids organized by mode
        result = {}
        all_pattern = r'AlignmentErrors_mrad/([^/]+)/[^/]+/([^/]+)'
        
        for tag in scalar_tags:
            match = re.search(all_pattern, tag)
            if match:
                mode = match.group(1)
                calib_id = match.group(2)
                
                if mode not in result:
                    result[mode] = set()
                
                result[mode].add(calib_id)
        
        # Convert sets to sorted lists
        for mode in result:
            result[mode] = sorted(list(result[mode]))
        
        return result

def get_alignment_errors(tensorboard_path, heliostat_id, mode=None, last_epoch_only=True):
    """
    Extract alignment errors for a specific heliostat
    
    Args:
        tensorboard_path: Path to the tensorboard log directory
        heliostat_id: Heliostat ID to extract data for
        mode: Optional, filter by mode (Train/Valid/Test)
        last_epoch_only: If True, only return the error at the last epoch
    
    Returns:
        DataFrame with alignment errors
    """
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    
    scalar_tags = ea.Tags()['scalars']
    
    # Filter tags for the specified heliostat
    if mode:
        pattern = f'AlignmentErrors_mrad/{mode}/{heliostat_id}/'
        heliostat_tags = [tag for tag in scalar_tags if tag.startswith(pattern)]
    else:
        pattern = f'AlignmentErrors_mrad/[^/]+/{heliostat_id}/'
        heliostat_tags = [tag for tag in scalar_tags if re.match(pattern, tag)]
    
    data = []
    
    for tag in heliostat_tags:
        events = ea.Scalars(tag)
        
        # Parse tag to extract mode and calib_id
        parts = tag.split('/')
        current_mode = parts[1]
        calib_id = parts[3]
        
        if last_epoch_only and events:
            # Get only the last epoch's data
            last_event = events[-1]
            data.append({
                'heliostat_id': heliostat_id,
                'mode': current_mode,
                'calib_id': calib_id,
                'epoch': last_event.step,
                'error': last_event.value
            })
        else:
            # Get all epochs' data
            for event in events:
                data.append({
                    'heliostat_id': heliostat_id,
                    'mode': current_mode,
                    'calib_id': calib_id,
                    'epoch': event.step,
                    'error': event.value
                })
    
    return pd.DataFrame(data)


def get_average_errors(tensorboard_path, heliostat_id):
    """
    Extract average alignment errors for a specific heliostat from Avg tags
    
    Args:
        tensorboard_path: Path to the tensorboard log directory
        heliostat_id: Heliostat ID to extract data for
    
    Returns:
        Dictionary with average errors by mode
    """
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    
    scalar_tags = ea.Tags()['scalars']
    
    # Look for average error tags
    pattern = f'AlignmentErrors_mrad/Avg/[^/]+/{heliostat_id}'
    avg_tags = [tag for tag in scalar_tags if re.match(pattern, tag)]
    
    avg_errors = {}
    
    for tag in avg_tags:
        events = ea.Scalars(tag)
        
        if events:
            # Get the last event (most recent epoch)
            last_event = events[-1]
            
            # Parse tag to extract mode
            # Format: AlignmentErrors_mrad/Avg/{mode}/{heliostat_id}
            parts = tag.split('/')
            mode = parts[2]
            
            avg_errors[mode] = {
                'error': last_event.value,
                'epoch': last_event.step
            }
    
    return avg_errors


def get_all_alignment_errors(tensorboard_path, last_epoch_only=True, heliostat_ids=[]):
    """
    Extract alignment errors for all heliostats
    
    Args:
        tensorboard_path: Path to the tensorboard log directory
        last_epoch_only: If True, only return the error at the last epoch
    
    Returns:
        DataFrame with alignment errors for all heliostats
    """
    # Get all heliostat IDs
    if len(heliostat_ids) == 0:  # if unknown, find in Tensorboard
        heliostat_ids = get_heliostat_ids(tensorboard_path)
    
    # Initialize empty list to store DataFrames
    dfs = []
    
    # Extract data for each heliostat
    for heliostat_id in heliostat_ids:
        df = get_alignment_errors(tensorboard_path, heliostat_id, last_epoch_only=last_epoch_only)
        dfs.append(df)
    
    # Combine all DataFrames
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# Example usage:
if __name__ == "__main__":
    # Replace with your tensorboard path
    tensorboard_path = "/dss/dsshome1/05/di38kid/data/results/tensorboard/run_2504011544_AA39.AC27.AD43.AM35.BB72.BG24"
    
    # Get all heliostat IDs
    # heliostat_ids = get_heliostat_ids(tensorboard_path)
    heliostat_ids = ['AA39', 'AC27', 'AD43', 'AM35', 'BB72', 'BG24']  # if known
    print(f"Found {len(heliostat_ids)} heliostats: {heliostat_ids}")
    
    # Get calibration IDs for all modes
    calib_ids_by_mode = get_calib_ids(tensorboard_path)
    for mode, calib_ids in calib_ids_by_mode.items():
        print(f"Mode {mode}: {len(calib_ids)} calibration IDs")
    
    # Get alignment errors for the first heliostat
    if heliostat_ids:
        first_heliostat = heliostat_ids[0]
        df = get_alignment_errors(tensorboard_path, first_heliostat)
        print(f"\nAlignment errors for heliostat {first_heliostat}:")
        print(df.head())
