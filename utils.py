import s3fs
import pandas as pd

def get_parquet_from_path(path):
    global BUCKET
    global PATH_TO_DATA
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}, anon=True)
    with fs.open(f"s3://{BUCKET}/{PATH_TO_DATA}/{path}") as f:
        return(pd.read_parquet(f))

