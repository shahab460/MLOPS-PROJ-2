import os
import pandas as pd
import gcsfs
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_names = self.config["bucket_file_names"]

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info("Data Ingestion Started...")

    def download_csv_from_gcp(self):
        try:
            logger.info(f"in download_csv_from_gcp: bucket name={self.bucket_name}")
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            # Initialize gcsfs for direct reading of large files
            fs = gcsfs.GCSFileSystem(project=client.project)

            logger.info(f"in download_csv_from_gcp: before for loop: bucket name={self.bucket_name}")
            
            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)
                gcs_path = f"gs://{self.bucket_name}/{file_name}"

                if file_name=="animelist.csv":
                    #blob = bucket.blob(file_name)
                    #blob.download_to_filename(file_path)
                    #data = pd.read_csv(file_path, nrows=5000000)
                    #data.to_csv(file_path, index=False)
                    data = pd.read_csv(gcs_path, nrows=5000000, storage_options={"project": client.project})
                    data.to_csv(file_path, index=False)

                    logger.info("Large file detected. Downloading only 5 million rows...")
                else:
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    logger.info("Downloading smaller files i.e anime and anime with synopsis...")
        except Exception as e:
            logger.error("Error while downloading data from GCP")
            raise CustomException("Failed to download data", e)

    def run(self):
        try:
            logger.info("Starting Data Ingestion Process ...")
            self.download_csv_from_gcp()
            logger.info("Data Ingestion Completed...")
        except CustomException as ce:
            logger.error(f"Error while running Data Ingestion Process : {str(ce)}")
        finally:
            logger.info("Data Ingestion DONE!")

if __name__=="__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()