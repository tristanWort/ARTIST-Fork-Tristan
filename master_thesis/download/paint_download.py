import argparse
from pathlib import Path
from paint.util import set_logger_config
import paint.util.paint_mappings as mappings
from paint.data.stac_client import StacClient


"""Script for downloading data from PAINT data from the STAC client."""
# TODO: Turn the script into a function or possibly class to manage data downloader.

set_logger_config()

# Data will be downloaded to this path.
download_path = Path(r"/jump/tw/data/paint/metadata/")

# Read in arguments for command line passing.
parser = argparse.ArgumentParser()

# Add arguments to the parser specifying the download output path.
parser.add_argument(
    "--output_dir",
    type=str,
    help="Path to save the downloaded data.",
    default=f"{download_path}",
)

# Add arguments to the parser specifying the weather data sources.
# parser.add_argument(
#     "--weather_data_sources",
#     type=str,
#     help="List of data sources to use for weather data.",
#     nargs="+",
#     choices=["Jülich", "DWD"],
#     default=["Jülich", "DWD"],
# )

# Add arguments to the parser specifying the start date for filtering the data.
# parser.add_argument(
#     "--start_date",
#     type=str,
#     help="End date for filtering the data.",
#     default="2023-01-01Z00:00:00Z",
# )

# Add arguments to the parser specifying the end date for filtering the data.
# parser.add_argument(
#     "--end_date",
#     type=str,
#     help="End data for filtering the date.",
#     default="2023-03-01Z00:00:00Z",
# )

# Add arguments to the parser specifying the heliostats to be downloaded.
parser.add_argument(
    "--heliostats",
    type=str,
    help="List of heliostats to be downloaded.",
    nargs="+",
    default=["AA39", "AM35"],
)

# Add arguments to the parser specifying the collections to be downloaded.
parser.add_argument(
    "--collections",
    type=str,
    help="List of collections to be downloaded.",
    nargs="+",
    choices=[
        mappings.SAVE_DEFLECTOMETRY.lower(),
        mappings.SAVE_CALIBRATION.lower(),
        mappings.SAVE_PROPERTIES.lower(),
    ],
    default=["deflectometry", "calibration", "properties"]
)

# Add arguments to the parser specifying the calibration items to download.
parser.add_argument(
    "--filtered_calibration",
    type=str,
    help="List of calibration items to download.",
    nargs="+",
    choices=[
        mappings.CALIBRATION_RAW_IMAGE_KEY,
        mappings.CALIBRATION_FLUX_IMAGE_KEY,
        mappings.CALIBRATION_FLUX_CENTERED_IMAGE_KEY,
        mappings.CALIBRATION_PROPERTIES_KEY,
        mappings.CALIBRATION_CROPPED_IMAGE_KEY,
    ],
    default=["calibration_properties", "flux_image"],
)

args = parser.parse_args()

# Create STAC client.
client = StacClient(output_dir=args.output_dir)

# Download the tower measurements.
# client.get_tower_measurements()

# Download the weather data within the given time period.
# client.get_weather_data(
#     data_sources=args.weather_data_sources,
#     start_date=datetime.strptime(args.start_date, mappings.TIME_FORMAT),
#     end_date=datetime.strptime(args.end_date, mappings.TIME_FORMAT),
# )

# Download heliostat data for the given heliostats, the specified collections and calibration items.
# client.get_heliostat_data(
#     heliostats=args.heliostats,
#     collections=args.collections,
#     filtered_calibration_keys=args.filtered_calibration,
# )

# Download metadata for all heliostats. Will not be used in this script.
# WARNING: Running the following command with 'heliostats=None' will take a very long time!
client.get_heliostat_metadata(heliostats=['AA39', 'AM35'])
