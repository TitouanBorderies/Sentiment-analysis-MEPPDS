import s3fs
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
BUCKET = os.environ.get("BUCKET", "")
endpoint = os.environ.get("ENDPOINT", "")
PATH_TO_DATA = os.environ.get("PATH_TO_DATA", "")


def get_parquet_from_path(path):
    """Load a Parquet file from an S3 bucket and return it as a pandas DataFrame."""
    global BUCKET
    global PATH_TO_DATA
    global endpoint
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint}, anon=True)
    with fs.open(f"s3://{BUCKET}/{PATH_TO_DATA}/{path}") as f:
        return pd.read_parquet(f)
