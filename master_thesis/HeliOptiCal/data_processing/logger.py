import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import logging
import seaborn as sns

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, repo_path)
from HeliOptiCal.utils.util import normalize_images_with_threshold

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
# A logger for the TensorboardLogger and TensorboardReader.


class TensorboardLogger:
    """
    A wrapper around SummaryWriter for more structured logging to TensorBoard.
    
    This class provides methods to log various types of data during training, including:
    - Losses
    - Metrics
    - Parameters
    - Images
    
    It organizes the logging in a consistent hierarchy for easier retrieval later.
    """
    
    def __init__(self, run: str, heliostat_names: List[str], log_dir: str = None, **kwargs):
        """
        Initialize the TensorboardLogger.
        
        Parameters
        ----------
        run : str
            Name of the run
        heliostat_names : List[str]
            List of Heliostat names.
        log_dir : str, optional
            Directory where logs will be saved.
        **kwargs
            Additional arguments to pass to SummaryWriter
        """
        if log_dir is None:
            log_dir = os.path.join('runs', name, 'log')
        
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.run = run
        self.writer = SummaryWriter(log_dir=log_dir/run, **kwargs)
        self.mode = "Train"  # Default mode
        self.heliostat_names = heliostat_names
        self.helio_and_calib_ids = dict()
        
        # DataFrames to accumulate logs for losses and metrics
        self.loss_logs = []  # List[Dict]
        self.metric_logs = []  # List[Dict]
        self.heliostat_metric_logs = defaultdict(list)  # Dict[metric_name, List[Dict]]
        log.info(f"TensorboardLogger initialized for run: {run}")
    
    def set_mode(self, mode: str):
        """
        Set current mode.
        
        Parameters
        ----------
        mode : str
            Mode of logging, eg. "Train", "Validation", "Test"
        """
        self.mode = mode
        
    def set_helio_and_calib_ids(self, helio_and_calib_ids: Dict[str, List[int]]):
        """
        Set mapping dictionary for Heliostat IDs to batch of calibration IDs.
        
        Parameters
        ----------
        helio_and_calib_ids : Dict[str, List[int]]
            Mapping from heliostat ID to list of calibration IDs (default is None, then use self.helio_and_calib_ids).
        """
        assert list(helio_and_calib_ids.keys()) == self.heliostat_names, \
            "New mapping for Heliostats to calibration IDs includes unexpected Heliostat names or has them in the wrong order."
        self.helio_and_calib_ids = helio_and_calib_ids
    
    def log_loss(self, loss_name: str, value: float, epoch: int):
        """
        Log a loss value.
        
        Parameters
        ----------
        loss_name : str
            Name of the loss (e.g., 'MSE', 'CHD', 'MAAE')
        value : float
            Loss value
        mode : str
            Training mode ('Train', 'Validation', 'Test')
        epoch : int
            Current epoch
        """
        assert self.mode is not None, "Mode cannot be None to log loss."
        self.writer.add_scalar(f"Loss/{loss_name}/{self.mode}", value, epoch)
        
        self.loss_logs.append({'epoch': epoch, 'name': loss_name, 'mode': self.mode, 'value': value})
            
    def log_metric(self, epoch: int, metric_name: str, value: float):
        """
        Log a metric value.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric (e.g., 'AlignmentError', 'SSIM')
        value : float
            Metric value
        epoch : int
            Current epoch
        """
        assert self.mode is not None, "Mode cannot be None to log metric."
        self.writer.add_scalar(f"Metrics/{metric_name}/{self.mode}", value, epoch)
        
        self.metric_logs.append({'epoch': epoch, 'name': metric_name, 'mode': self.mode, 'value': value})
    
    def log_heliostat_metric(self, epoch: int, metric_name: str, heliostat_id: str, value: float, calib_id: int = None):
        """
        Log a metric value for a specific heliostat.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric (e.g., 'AlignmentError', 'SSIM')
        heliostat_id : str
            ID of the heliostat
        value : float
            Metric value
        epoch : int
            Current epoch
        calib_id : int, optional
            Calibration ID for the specific data point
        """
        assert self.mode is not None, "Mode cannot be None to log metric."
        assert heliostat_id in self.heliostat_names, "No match for this Heliostat ID."
        if calib_id is not None:
            self.writer.add_scalar(f"{metric_name}/{self.mode}/{heliostat_id}/{calib_id}", value, epoch)
        else:
            self.writer.add_scalar(f"{metric_name}/{self.mode}/{heliostat_id}", value, epoch)
            
        log_entry = {'epoch': epoch, 'mode': self.mode, 'heliostat_id': heliostat_id, 'value': value}
        if calib_id is not None:
            log_entry['calibration_id'] = calib_id
        self.heliostat_metric_logs[metric_name].append(log_entry)

    def log_parameter_as_scalar(self, epoch: int, param_name: str, heliostat_id: str, value: torch.Tensor, index: int = None):
        """
        Log a parameter value as a scalar.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter
        heliostat_id : str
            ID of the heliostat
        value : torch.Tensor
            Parameter value
        epoch : int
            Current epoch
        index : int, optional
            Index for the parameter if it's part of a list
        """
        assert heliostat_id in self.heliostat_names, "No match for this Heliostat ID."
        if index is not None:
            self.writer.add_scalar(f"Parameters/{param_name}/{heliostat_id}/{index}", value.item(), epoch)
        else:
            self.writer.add_scalar(f"Parameters/{param_name}/{heliostat_id}", value.item(), epoch)
    
    def log_parameter_as_histogram(self, epoch: int, param_name: str, heliostat_id: str, value: torch.Tensor, index: int = None):
        """
        Log a parameter value as a histogram.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter
        heliostat_id : str
            ID of the heliostat
        value : torch.Tensor
            Parameter value
        epoch : int
            Current epoch
        """
        if index is not None:
            self.writer.add_histogramm(f"Parameters/{param_name}/{heliostat_id}/{index}", value.item(), epoch)
        else:
            self.writer.add_histogramm(f"Parameters/{param_name}/{heliostat_id}", value.item(), epoch)
    
    def log_parameters_obj(self, epoch, obj, name=None, heliostat_idx=None, index=None):
        
        with torch.no_grad():
            # Handle the previous recursive logic for backward compatibility
            if isinstance(obj, torch.nn.Parameter):
                heliostat_name = self.heliostat_names[heliostat_idx]
                self.log_parameter_as_scalar(epoch, name, heliostat_name, obj, index)
                
            elif isinstance(obj, (torch.nn.ParameterList, List, Tuple)):
                if heliostat_idx is None:
                    if len(obj) == len(self.heliostat_names):
                        for h, param in enumerate(obj):
                            self.log_parameters_obj(epoch=epoch, obj=param, name=name, heliostat_idx=h)
                    else:
                        for i, param in enumerate(obj):
                            self.log_parameters_obj(epoch=epoch, obj=param, name=name, heliostat_idx=heliostat_idx, index=i)
                    
                else: 
                    [self.log_parameters_obj(epoch=epoch, obj=param, name=name, heliostat_idx=heliostat_idx, index=i) 
                    for i, param in enumerate(obj)]

            elif isinstance(obj, (torch.nn.ParameterDict, Dict)):
                for name, obj in obj.items():
                    self.log_parameters_obj(epoch=epoch, obj=obj, name=name)
                    
            else:
                raise TypeError(f"Cannot log parameter of unsupported type: {type(obj)}")
    
    def log_parameters(self, epoch: int, obj: Union[Dict, List, torch.nn.Parameter], heliostat_names=None, use_scalars: bool = True):
        """
        Log all parameters from a parameter dictionary.
        
        Parameters
        ----------
        params_dict : Dict[str, torch.nn.Parameter]
            Dictionary of parameters
        heliostat_names : List[str]
            List of heliostat names (default is None, then use self.helio_and_calib_ids)
        epoch : int
            Current epoch
        use_scalars : bool, optional
            Whether to log parameters as scalars (True) or histograms (False)
        """
        if heliostat_names is None:
            heliostat_names = self.heliostat_names
        
        with torch.no_grad():
            for param_name, param in params_dict.items(): 
                
                if isinstance(param, (Tuple, List, torch.nn.ParameterList)):
                    assert len(param) == len(heliostat_names), \
                        f"Parameter list length mismatch: {len(param)} vs {len(heliostat_names)}"
                    for h_idx, (h_name, param_value) in enumerate(zip(heliostat_names, param)):
                        if use_scalars:
                            self.log_parameter_as_scalar(epoch, param_name, h_name, param_value)
 
                        else:
                            if param_value.numel() == 1:
                                self.log_parameter_as_histogram(epoch, param_name, h_name, param_value)
                            else:
                                # For multi-element parameters, log each element as a separate scalar
                                for i, val in enumerate(param_value.flatten()):
                                    self.log_parameter_as_histogram(epoch, param_name, h_name, val, index=i)
                            
                elif isinstance(param, torch.nn.Parameter):
                    assert param.shape[0] == len(heliostat_names), \
                        f"Parameter batch dimension mismatch: {param.shape[0]} vs {len(heliostat_names)}"
                    
                    for h_idx, (h_name, param_value) in enumerate(zip(heliostat_names, list(param))):
                        param_value = param_value
                        if use_scalars:
                            if param_value.numel() == 1:
                                self.log_parameter_as_scalar(epoch, param_name, h_name, param_value)
                            else:
                                # For multi-element parameters, log each element as a separate scalar
                                for i, val in enumerate(param_value.flatten()):
                                    self.log_parameter_as_scalar(epoch, param_name, h_name, val, index=i)
                        else:
                            if param_value.numel() == 1:
                                self.log_parameter_as_histogram(epoch, param_name, h_name, param_value)
                            else:
                                # For multi-element parameters, log each element as a separate scalar
                                for i, val in enumerate(param_value.flatten()):
                                    self.log_parameter_as_histogram(epoch, param_name, h_name, val, index=i)
                
                else:
                    raise TypeError(f"Cannot log parameter of unsupported type: {type(param)}")
    
    def log_alignment_errors(self, epoch: int, alignment_errors: torch.Tensor, helio_and_calib_ids=None, is_actual=False):
        """
        Log alignment errors at batch level, heliostat level, and individual calibration level.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        alignment_errors : torch.Tensor
            Tensor of alignment errors with shape [B, H] in mrad.
        helio_and_calib_ids : Dict[str, List[int]]
            Mapping from heliostat ID to list of calibration IDs (default is None, then use self.helio_and_calib_ids).
        is_actual : bool
            Indication whether the alignment errors are the "actual" alignment errors.
        """
        if helio_and_calib_ids is None:
            helio_and_calib_ids = self.helio_and_calib_ids
            
        if self.mode is None:
            raise RuntimeError("Logger mode must be set before logging.")

        assert alignment_errors.dim() == 2, f"Expected alignment_erorrs to have 2 dimensions, not {alignment_errors.dim()}"
        alignment_errors = alignment_errors.detach().cpu()
        
        assert alignment_errors.shape[1] == len(helio_and_calib_ids)
        
        base_name = "AlignmentErrors_mrad"
        if is_actual:
            base_name = "ActualAlignmentErrors_mrad"
        
        # Summary metrics
        self.log_metric(epoch, f"{base_name}/Avg", alignment_errors.mean().item())
        self.log_metric(epoch, f"{base_name}/Med", alignment_errors.median().item())

        # Per-heliostat logging
        for h_idx, (helio_id, calib_ids) in enumerate(helio_and_calib_ids.items()):
            helio_errors = alignment_errors[:, h_idx] if alignment_errors.shape[1] == len(helio_and_calib_ids) else alignment_errors[h_idx]
            self.log_heliostat_metric(epoch, f"{base_name}/Avg", helio_id, helio_errors.mean().item())
            self.log_heliostat_metric(epoch, f"{base_name}/Med", helio_id, helio_errors.median().item())

            for c_idx, (calib_id, err) in enumerate(zip(calib_ids, helio_errors.tolist())):
                self.log_heliostat_metric(epoch, f"{base_name}", helio_id, err, calib_id=calib_id)
    
    def log_image(self, epoch: int, image: torch.Tensor, image_name: str, heliostat_id: str, calib_id: int=None):
        """
        Log an image.
        
        Parameters
        ----------
        image_name : str
            Name of the image (e.g., 'FluxPrediction', 'FluxDiff')
        heliostat_id : str
            ID of the heliostat
        image : torch.Tensor
            Image tensor with shape [H, W]
        mode : str
            Training mode ('Train', 'Validation', 'Test')
        epoch : int
            Current epoch
        calib_id : int, optional
            Calibration ID for the specific data point
        """
        assert self.mode is not None, "Mode cannot be None to log image."
        if calib_id is not None:
            self.writer.add_image(f"{image_name}/{self.mode}/{heliostat_id}/{calib_id}", image.cpu().detach(), epoch, dataformats='HW')
        else:
            self.writer.add_image(f"{image_name}/{self.mode}/{heliostat_id}", image.cpu().detach(), epoch, dataformats='HW')
    
    def log_flux_bitmaps(self, epoch: int, bitmaps: torch.Tensor, type: str, helio_and_calib_ids=None, normalize=True):
        """
        Log a batch of flux bitmaps per heliostat and calibration ID.

        Parameters
        ----------
        bitmaps : torch.Tensor
            Tensor of shape [B, H, H_img, W_img], containing image bitmaps.
        epoch : int
            Current epoch.
        helio_and_calib_ids : Dict[str, List[int]]
            Dictionary mapping heliostat IDs to calibration ID lists (default is None, then use self.helio_and_calib_ids).
        """
        if helio_and_calib_ids is None:
            helio_and_calib_ids = self.helio_and_calib_ids
            
        if self.mode is None:
            raise RuntimeError("Logger mode must be set before logging.")

        bitmaps = bitmaps.detach().cpu()
        B, H, H_img, W_img = bitmaps.shape
        
        norm_bitmaps = torch.zeros_like(bitmaps)
        if normalize:
            for b in range(B):
                norm_bitmaps[b] = normalize_images_with_threshold(bitmaps[b], threshold=1)
        else:
            norm_bitmaps = bitmaps

        for h_idx, (heliostat_id, calib_ids) in enumerate(helio_and_calib_ids.items()):
            for b in range(len(calib_ids)):
                calib_id = calib_ids[b]
                image = norm_bitmaps[b, h_idx]  # shape [H_img, W_img]
                self.log_image(epoch, image, type, heliostat_id, calib_id=calib_id)
                
    def log_diff_flux_bitmaps(self, epoch: int, pred_bitmaps: torch.Tensor, true_bitmaps: torch.Tensor, type: str, helio_and_calib_ids=None, normalize=True):
        """
        Log a batch of flux bitmaps per heliostat and calibration ID.

        Parameters
        ----------
        bitmaps : torch.Tensor
            Tensor of shape [B, H, H_img, W_img], containing image bitmaps.
        epoch : int
            Current epoch.
        helio_and_calib_ids : Dict[str, List[int]]
            Dictionary mapping heliostat IDs to calibration ID lists (default is None, then use self.helio_and_calib_ids).
        """
        assert pred_bitmaps.shape == true_bitmaps.shape, "pred_bitmaps and true_bitmaps must have equal shapes."
        
        if helio_and_calib_ids is None:
            helio_and_calib_ids = self.helio_and_calib_ids
            
        if self.mode is None:
            raise RuntimeError("Logger mode must be set before logging.")

        pred_bitmaps = pred_bitmaps.detach().cpu()
        true_bitmaps = true_bitmaps.detach().cpu()
        B, H, H_img, W_img = pred_bitmaps.shape
        
        norm_diff_bitmaps = torch.zeros_like(pred_bitmaps)
        if normalize:
            for b in range(B):
                norm_pred_bitmaps = normalize_images_with_threshold(pred_bitmaps[b], threshold=1)
                norm_true_bitmaps = normalize_images_with_threshold(true_bitmaps[b], threshold=1)
                norm_diff_bitmaps[b] = norm_pred_bitmaps - norm_true_bitmaps
        else:
            norm_diff_bitmaps = pred_bitmaps - true_bitmaps

        for h_idx, (heliostat_id, calib_ids) in enumerate(helio_and_calib_ids.items()):
            for b in range(len(calib_ids)):
                calib_id = calib_ids[b]
                image = norm_diff_bitmaps[b, h_idx]  # shape [H_img, W_img]
                self.log_image(epoch, image, type, heliostat_id, calib_id=calib_id)
                
    def save_dataframes_to_csv(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Save all logged losses and metrics to CSV files.

        Parameters
        ----------
        output_dir : Union[str, Path]
            Path to directory where CSV files will be saved.
        """
        if output_dir is None:
            output_dir = self.log_dir

        os.makedirs(output_dir, exist_ok=True)
            
        # Losses
        if self.loss_logs:
            df_loss = pd.DataFrame(self.loss_logs)
            df_loss_pivot = df_loss.pivot_table(index='epoch', columns=['name', 'mode'], values='value')
            df_loss_pivot.to_csv(os.path.join(output_dir, "losses.csv"))

        # Metrics
        if self.metric_logs:
            df_metric = pd.DataFrame(self.metric_logs)
            df_metric_pivot = df_metric.pivot_table(index='epoch', columns=['name', 'mode'], values='value')
            df_metric_pivot.to_csv(os.path.join(output_dir, "metrics.csv"))

        # Heliostat-specific metrics
        for metric_name, records in self.heliostat_metric_logs.items():
            df = pd.DataFrame(records)

            if 'calibration_id' in df.columns:
                col_index = ['heliostat_id', 'mode', 'calibration_id']
            else:
                col_index = ['heliostat_id', 'mode']

            df_pivot = df.pivot_table(index='epoch', columns=col_index, values='value')
            file_path = os.path.join(output_dir, f"{metric_name}.csv")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df_pivot.to_csv(file_path)
            
        log.info("Saved Metrics to csv-files at location: {self.output_idr}")

    def close(self):
        """Close the SummaryWriter."""
        self.writer.close()
        self.save_dataframes_to_csv()
        log.info(f"TensorboardLogger closed: {self.run}")


class TensorboardReader:
    """
    A class for reading and analyzing TensorBoard logs.
    
    This class provides methods to read TensorBoard event files, extract data
    into pandas DataFrames, and visualize the data.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize the TensorboardReader.
        
        Parameters
        ----------
        log_dir : str
            Directory where logs are saved
        """
        self.log_dir = log_dir
        self.event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
        if not self.event_files:
            raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
        
        self.event_acc = EventAccumulator(self.event_files[0])
        self.event_acc.Reload()
        log.info(f"TensorboardReader initialized: {log_dir}")
        
        # Cache for loaded data
        self._scalar_cache = {}
        self._histogram_cache = {}
        self._image_cache = {}
    
    def get_tags(self, data_type: str = 'scalars') -> List[str]:
        """
        Get all tags for a specific data type.
        
        Parameters
        ----------
        data_type : str, optional
            Type of data to get tags for ('scalars', 'histograms', 'images')
            
        Returns
        -------
        List[str]
            List of tags
        """
        if data_type == 'scalars':
            return self.event_acc.Tags()['scalars']
        elif data_type == 'histograms':
            return self.event_acc.Tags()['histograms']
        elif data_type == 'images':
            return self.event_acc.Tags()['images']
        else:
            raise ValueError(f"Invalid data_type: {data_type}")
    
    def get_scalar_data(self, tag: str) -> pd.DataFrame:
        """
        Get scalar data for a specific tag.
        
        Parameters
        ----------
        tag : str
            Tag to get data for
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['wall_time', 'step', 'value']
        """
        if tag not in self._scalar_cache:
            events = self.event_acc.Scalars(tag)
            self._scalar_cache[tag] = pd.DataFrame(
                [(e.wall_time, e.step, e.value) for e in events],
                columns=['wall_time', 'step', 'value']
            )
        return self._scalar_cache[tag]
    
    def get_histogram_data(self, tag: str) -> Dict[int, np.ndarray]:
        """
        Get histogram data for a specific tag.
        
        Parameters
        ----------
        tag : str
            Tag to get data for
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping steps to histogram data
        """
        if tag not in self._histogram_cache:
            events = self.event_acc.Histograms(tag)
            self._histogram_cache[tag] = {
                e.step: np.array(e.histogram_value.bucket_limit[:-1]) 
                for e in events
            }
        return self._histogram_cache[tag]
    
    def get_image_data(self, tag: str, step: int = 0) -> np.ndarray:
        """
        Get image data for a specific tag and step.
        
        Parameters
        ----------
        tag : str
            Tag to get data for
        step : int, optional
            Step to get data for
            
        Returns
        -------
        np.ndarray
            Image data
        """
        cache_key = f"{tag}_{step}"
        if cache_key not in self._image_cache:
            events = self.event_acc.Images(tag)
            for e in events:
                if e.step == step:
                    self._image_cache[cache_key] = e.encoded_image_string
                    break
        return self._image_cache[cache_key]
    
    def get_loss_data(self, loss_name: str = None, mode: str = None) -> pd.DataFrame:
        """
        Get loss data, optionally filtered by loss name and mode.
        
        Parameters
        ----------
        loss_name : str, optional
            Name of the loss to filter by
        mode : str, optional
            Mode to filter by ('Train', 'Validation', 'Test')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with loss data
        """
        loss_tags = [tag for tag in self.get_tags() if tag.startswith('Loss/')]
        
        if loss_name:
            loss_tags = [tag for tag in loss_tags if f"/{loss_name}/" in tag]
        
        if mode:
            loss_tags = [tag for tag in loss_tags if f"/{mode}" in tag]
        
        data = []
        for tag in loss_tags:
            df = self.get_scalar_data(tag)
            parts = tag.split('/')
            
            # Extract loss name and mode from tag
            if len(parts) >= 3:
                current_loss_name = parts[1]
                current_mode = parts[2]
            else:
                current_loss_name = 'unknown'
                current_mode = 'unknown'
            
            df['loss_name'] = current_loss_name
            df['mode'] = current_mode
            data.append(df)
        
        if not data:
            return pd.DataFrame()
        
        return pd.concat(data, ignore_index=True)
    
    def get_metric_data(self, metric_name: str = None, mode: str = None, 
                        heliostat_id: str = None) -> pd.DataFrame:
        """
        Get metric data, optionally filtered by metric name, mode, and heliostat ID.
        
        Parameters
        ----------
        metric_name : str, optional
            Name of the metric to filter by
        mode : str, optional
            Mode to filter by ('Train', 'Validation', 'Test')
        heliostat_id : str, optional
            Heliostat ID to filter by
            
        Returns
        -------
        pd.DataFrame
            DataFrame with metric data
        """
        # Get all tags that match the metric name pattern
        if metric_name:
            metric_tags = [tag for tag in self.get_tags() if tag.startswith(f"{metric_name}/")]
        else:
            # Exclude Loss/ tags which are handled separately
            metric_tags = [tag for tag in self.get_tags() 
                          if not tag.startswith('Loss/') and not tag.startswith('Parameters/')]
        
        if mode:
            metric_tags = [tag for tag in metric_tags if f"/{mode}/" in tag]
        
        if heliostat_id:
            metric_tags = [tag for tag in metric_tags if f"/{heliostat_id}" in tag or f"/{heliostat_id}/" in tag]
        
        data = []
        for tag in metric_tags:
            df = self.get_scalar_data(tag)
            parts = tag.split('/')
            
            # Extract metric information from tag
            if len(parts) >= 2:
                current_metric = parts[0]
            else:
                current_metric = 'unknown'
                
            if len(parts) >= 3:
                if parts[1] == 'Avg':
                    current_mode = parts[2]
                    current_is_avg = True
                else:
                    current_mode = parts[1]
                    current_is_avg = False
            else:
                current_mode = 'unknown'
                current_is_avg = False
            
            if len(parts) >= 4:
                if current_is_avg:
                    current_heliostat = parts[3]
                    current_calib_id = None
                else:
                    current_heliostat = parts[2]
                    if len(parts) >= 5:
                        try:
                            current_calib_id = int(parts[3])
                        except ValueError:
                            current_calib_id = parts[3]
                    else:
                        current_calib_id = None
            else:
                current_heliostat = None
                current_calib_id = None
            
            df['metric'] = current_metric
            df['mode'] = current_mode
            df['heliostat_id'] = current_heliostat
            df['calib_id'] = current_calib_id
            df['is_average'] = current_is_avg
            data.append(df)
        
        if not data:
            return pd.DataFrame()
        
        return pd.concat(data, ignore_index=True)
    
    def get_parameter_data(self, param_name: str = None, 
                          heliostat_id: str = None) -> pd.DataFrame:
        """
        Get parameter data, optionally filtered by parameter name and heliostat ID.
        
        Parameters
        ----------
        param_name : str, optional
            Name of the parameter to filter by
        heliostat_id : str, optional
            Heliostat ID to filter by
            
        Returns
        -------
        pd.DataFrame
            DataFrame with parameter data
        """
        # Get all tags that match the parameter pattern
        param_tags = [tag for tag in self.get_tags() if tag.startswith('Parameters/')]
        
        if param_name:
            param_tags = [tag for tag in param_tags if f"Parameters/{param_name}/" in tag]
        
        if heliostat_id:
            param_tags = [tag for tag in param_tags if f"/{heliostat_id}" in tag or f"/{heliostat_id}/" in tag]
        
        data = []
        for tag in param_tags:
            df = self.get_scalar_data(tag)
            parts = tag.split('/')
            
            # Extract parameter information from tag
            if len(parts) >= 3:
                current_param = parts[1]
                current_heliostat = parts[2]
            else:
                current_param = 'unknown'
                current_heliostat = 'unknown'
            
            index = None
            if len(parts) >= 4:
                try:
                    index = int(parts[3])
                except ValueError:
                    pass
            
            df['param_name'] = current_param
            df['heliostat_id'] = current_heliostat
            df['index'] = index
            data.append(df)
        
        if not data:
            return pd.DataFrame()
        
        return pd.concat(data, ignore_index=True)
    
    def plot_losses(self, loss_names: List[str] = None, modes: List[str] = None, 
                   figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot loss curves.
        
        Parameters
        ----------
        loss_names : List[str], optional
            Names of losses to plot
        modes : List[str], optional
            Modes to plot ('Train', 'Validation', 'Test')
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        # Get loss data
        df = self.get_loss_data()
        if df.empty:
            return plt.figure(figsize=figsize)
        
        if loss_names:
            df = df[df['loss_name'].isin(loss_names)]
        
        if modes:
            df = df[df['mode'].isin(modes)]
        
        if df.empty:
            return plt.figure(figsize=figsize)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for (loss_name, mode), group_df in df.groupby(['loss_name', 'mode']):
            ax.plot(group_df['step'], group_df['value'], label=f"{loss_name} - {mode}")
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_metrics(self, metric_name: str, modes: List[str] = None, 
                    heliostat_ids: List[str] = None, figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot metric curves.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to plot
        modes : List[str], optional
            Modes to plot ('Train', 'Validation', 'Test')
        heliostat_ids : List[str], optional
            Heliostat IDs to plot
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        # Get metric data
        df = self.get_metric_data(metric_name=metric_name)
        if df.empty:
            return plt.figure(figsize=figsize)
        
        if modes:
            df = df[df['mode'].isin(modes)]
        
        if heliostat_ids:
            df = df[df['heliostat_id'].isin(heliostat_ids)]
        
        if df.empty:
            return plt.figure(figsize=figsize)
        
        # Plot average metrics if available
        avg_df = df[df['is_average'] == True]
        non_avg_df = df[df['is_average'] == False]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot average metrics
        if not avg_df.empty:
            for (mode, heliostat_id), group_df in avg_df.groupby(['mode', 'heliostat_id']):
                label = f"Avg {heliostat_id} - {self.mode}" if heliostat_id else f"Avg - {self.mode}"
                ax.plot(group_df['step'], group_df['value'], 
                        label=label, linewidth=2)
        
        # Plot individual metrics if there aren't too many
        if not non_avg_df.empty and len(non_avg_df['calib_id'].unique()) <= 10:
            for (mode, heliostat_id, calib_id), group_df in non_avg_df.groupby(['mode', 'heliostat_id', 'calib_id']):
                if pd.isna(calib_id):
                    continue
                label = f"{heliostat_id} (ID:{calib_id}) - {self.mode}"
                ax.plot(group_df['step'], group_df['value'], 
                        label=label, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{metric_name} Value')
        ax.set_title(f'{metric_name} Curves')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_parameters(self, param_name: str, heliostat_ids: List[str] = None,
                       figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot parameter evolution.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter to plot
        heliostat_ids : List[str], optional
            Heliostat IDs to plot
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        # Get parameter data
        df = self.get_parameter_data(param_name=param_name)
        if df.empty:
            return plt.figure(figsize=figsize)
        
        if heliostat_ids:
            df = df[df['heliostat_id'].isin(heliostat_ids)]
        
        if df.empty:
            return plt.figure(figsize=figsize)
        
        # Check if there are parameters with indices
        has_indices = not df['index'].isna().all()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if has_indices:
            # Group by heliostat and index
            for (heliostat_id, index), group_df in df.groupby(['heliostat_id', 'index']):
                if pd.isna(index):
                    label = f"{heliostat_id}"
                else:
                    label = f"{heliostat_id} [{index}]"
                ax.plot(group_df['step'], group_df['value'], label=label)
        else:
            # Group by heliostat only
            for heliostat_id, group_df in df.groupby('heliostat_id'):
                ax.plot(group_df['step'], group_df['value'], label=heliostat_id)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{param_name} Value')
        ax.set_title(f'{param_name} Evolution')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def visualize_final_error_distribution(self, metric_name: str = "AlignmentErrors_mrad",
                                         mode: str = "Test", figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Visualize the final error distribution across all heliostats.
        
        Parameters
        ----------
        metric_name : str, optional
            Name of the metric to visualize
        mode : str, optional
            Mode to visualize ('Train', 'Validation', 'Test')
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        # Get metric data
        df = self.get_metric_data(metric_name=metric_name, mode=mode)
        if df.empty:
            return plt.figure(figsize=figsize)
        
        # Keep only data points with valid heliostat_id and calib_id
        df = df[(~df['heliostat_id'].isna()) & (~df['calib_id'].isna()) & (df['is_average'] == False)]
        
        if df.empty:
            return plt.figure(figsize=figsize)
        
        # Get the last step for each heliostat and calibration ID
        max_step = df['step'].max()
        final_df = df[df['step'] == max_step]
        
        if final_df.empty:
            return plt.figure(figsize=figsize)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot by heliostat
        sns.boxplot(x='heliostat_id', y='value', data=final_df, ax=ax1)
        ax1.set_title(f'Final {metric_name} Distribution by Heliostat')
        ax1.set_xlabel('Heliostat ID')
        ax1.set_ylabel(f'{metric_name}')
        ax1.tick_params(axis='x', rotation=45)
        
        # Overall distribution
        sns.histplot(final_df['value'], kde=True, ax=ax2)
        ax2.set_title(f'Overall {metric_name} Distribution')
        ax2.set_xlabel(f'{metric_name}')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        return fig
    
    def create_performance_summary(self, metrics: List[str] = None, 
                                 mode: str = "Test") -> pd.DataFrame:
        """
        Create a summary of the final performance metrics.
        
        Parameters
        ----------
        metrics : List[str], optional
            Metrics to include in the summary
        mode : str, optional
            Mode to summarize ('Train', 'Validation', 'Test')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with performance summary
        """
        if metrics is None:
            # Try to find common metrics
            all_tags = self.get_tags()
            possible_metrics = []
            for tag in all_tags:
                parts = tag.split('/')
                if len(parts) > 1 and parts[0] not in ['Loss', 'Parameters']:
                    possible_metrics.append(parts[0])
            
            metrics = list(set(possible_metrics))
        
        summary_data = []
        
        for metric in metrics:
            df = self.get_metric_data(metric_name=metric, mode=mode)
            if df.empty:
                continue
            
            # Get average metrics
            avg_df = df[df['is_average'] == True]
            if avg_df.empty:
                continue
            
            # Get the last step
            max_step = avg_df['step'].max()
            final_avg_df = avg_df[avg_df['step'] == max_step]
            
            # Calculate statistics
            for heliostat_id, group_df in final_avg_df.groupby('heliostat_id'):
                summary_data.append({
                    'Metric': metric,
                    'Heliostat': heliostat_id,
                    'Value': group_df['value'].mean(),
                    'Mode': mode
                })
        
        if not summary_data:
            return pd.DataFrame()
        
        return pd.DataFrame(summary_data)

    def compare_configurations(self, other_readers: List['TensorboardReader'], 
                              metric: str, mode: str = "Test") -> Figure:
        """
        Compare performance across different configurations.
        
        Parameters
        ----------
        other_readers : List[TensorboardReader]
            List of other TensorboardReader instances to compare with
        metric : str
            Metric to compare
        mode : str, optional
            Mode to compare ('Train', 'Validation', 'Test')
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        readers = [self] + other_readers
        all_data = []
        
        for i, reader in enumerate(readers):
            df = reader.get_metric_data(metric_name=metric, mode=mode)
            if df.empty:
                continue
            
            # Keep only average metrics
            df = df[df['is_average'] == True]
            if df.empty:
                continue
            
            df['Configuration'] = f"Config {i+1}"
            all_data.append(df)
        
        if not all_data:
            return plt.figure()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for config, group_df in combined_df.groupby('Configuration'):
            ax.plot(group_df['step'], group_df['value'], label=config)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{metric} Value')
        ax.set_title(f'Comparison of {metric} Across Configurations')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig

if __name__ == '__main__':
    from util import analyze_calibration_results
    save_dir = '/dss/dsshome1/05/di38kid/data/results/runs/run_2504091425_AM35'
    analyze_calibration_results(save_dir=save_dir)
    