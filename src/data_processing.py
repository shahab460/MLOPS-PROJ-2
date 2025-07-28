import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
import sys

logger = get_logger(__name__)

    
def getAnimeName(anime_id, df):
    try:
        matching_row = df[df.anime_id == anime_id]
        if matching_row.empty:
            return "Anime not found"  # or some default value like "Anime not found"
        name = matching_row['eng_version'].values[0]
        if pd.isna(name):
            name = matching_row['Name'].values[0]
        return name
    except Exception as e:
        print(f"Error: {e}")
        return "Anime not found"  # or some default value

class DataProcessor():
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir

        self.rating_df = None
        self.anime_df = None
        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None

        self.user2user_encoded = {}
        self.user2user_decoded = {}
        self.anime2anime_encoded = {}
        self.anime2anime_decoded = {}

        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Data Processing Initialized ...")

    def load_data(self, usecols):
        try:
            self.rating_df = pd.read_csv(self.input_file, low_memory=True, usecols=usecols)
            logger.info("Data Loaded Successfully")

        except Exception as e:
            raise CustomException("Failed to load data", sys)
        
    def filter_users(self, min_rating=400):
        try:
            n_ratings = self.rating_df["user_id"].value_counts()
            self.rating_df = self.rating_df[self.rating_df["user_id"].isin(n_ratings[n_ratings >= 400].index)].copy()
            logger.info("Filtered Data Successfully ...")

        except Exception as e:
            raise CustomException("Error during filter user process ...", sys)

    def scale_ratings(self):
        try:
            min_rating = min(self.rating_df["rating"])
            max_rating = max(self.rating_df["rating"])
            avg_rating = np.mean(self.rating_df["rating"])

            self.rating_df["rating"] = self.rating_df["rating"].apply(lambda x: (x - min_rating)/(max_rating - min_rating)).values.astype(np.float64)
            logger.info("Scaling done successfully ...")

        except Exception as e:
            raise CustomException("Error during scaling ...", sys)
        
    def encode_data(self):
        try:
            ## users
            user_ids = self.rating_df["user_id"].unique().tolist()
            self.user2user_encoded = {x : i for i, x in enumerate(user_ids)}
            self.user2user_decoded = {i : x for i, x in enumerate(user_ids)}
            self.rating_df["user"] = self.rating_df["user_id"].map(self.user2user_encoded)

            ## anime
            anime_ids = self.rating_df["anime_id"].unique().tolist()
            self.anime2anime_encoded = {x : i for i, x in enumerate(anime_ids)}
            self.anime2anime_decoded = {i : x for i, x in enumerate(anime_ids)}
            self.rating_df["anime"] = self.rating_df["anime_id"].map(self.anime2anime_encoded)

            logger.info("Encoding done for users and anime successfully ...")
        except Exception as e:
            raise CustomException("Error during encoding data ...", sys)
        
    def split_data(self, test_size=1000):
        try:
            self.rating_df = self.rating_df.sample(frac=1, random_state=43).reset_index(drop=True)
            X = self.rating_df[["user", "anime"]].values
            y = self.rating_df["rating"]

            train_indices = self.rating_df.shape[0] - test_size
            X_train, X_test, y_train, y_test = (
                X[:train_indices],
                X[train_indices:],
                y[:train_indices],
                y[train_indices:]
            )
            
            self.X_train_array = [X_train[: , 0], X_train[: , 1]]
            self.X_test_array = [X_test[: , 0], X_test[: , 1]]
            self.y_test = y_test
            self.y_train = y_train

            ## just for debugging ##
            print("Checking for None in X_train:", np.any(self.X_train_array == None))
            print("Checking for None in y_train:", np.any(y_train == None))

            print("X_train shape:", np.shape(self.X_train_array))
            print("y_train shape:", np.shape(y_train))
            print("y_train content:", y_train)
            ## end debugging ##

            logger.info("Splitting data done successfully ...")
        except Exception as e:
            raise CustomException("Error during splitting data ...", sys)
        
    def save_artifacts(self):
        try:
            artifacts = {
                "user2user_encoded" : self.user2user_encoded,
                "user2user_decoded" : self.user2user_decoded,
                "anime2anime_encoded" : self.anime2anime_encoded,
                "anime2anime_decoded" : self.anime2anime_decoded
            }

            for name, data in artifacts.items():
                joblib.dump(data, os.path.join(self.output_dir, f"{name}.pkl"))
                logger.info(f"{name} saved successfully at artifacts directory ...")

            joblib.dump(self.X_train_array, X_TRAIN_ARRAY)
            joblib.dump(self.X_test_array, X_TEST_ARRAY)
            joblib.dump(self.y_train, Y_TRAIN)
            joblib.dump(self.y_test, Y_TEST)

            self.rating_df.to_csv(RATING_DF, index=False)

            logger.info("All training, testing and rating df data is saved successfully ...")
            
        except Exception as e:
            raise CustomException("Error during saving artifacts ...", sys)

        
    def process_anime_data(self):
        try:
            df = pd.read_csv(ANIME_CSV)
            cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
            synopsis_df = pd.read_csv(ANIME_WITH_SYNOPSIS, usecols=cols)

            df = df.replace("Unknown", np.nan)

            ## invoke getAnimeName function

            df["anime_id"] = df["MAL_ID"]
            # Directly compute eng_version using a vectorized approach
            df["eng_version"] = df["English name"].where(df["English name"].notna(), df["Name"])
            df["eng_version"] = df.anime_id.apply(lambda x:getAnimeName(x, df))

            df.sort_values(by=["Score"], inplace=True, ascending=False, kind="quicksort", na_position="last")

            df = df[["anime_id", "eng_version", "Score", "Genres", "Type", "Episodes", "Premiered", "Members"]]

            df.to_csv(DF_PATH, index=False)
            synopsis_df.to_csv(SYNOPSIS_DF, index=False)
            
            logger.info("Processed anime data, saved df and synopsis df successfully ...")
            
        except Exception as e:
            raise CustomException("Error during Process Anime Data ...", sys)
        
    def run(self):
        try:
            self.load_data(usecols=["user_id", "anime_id", "rating"])
            self.filter_users()
            self.scale_ratings()
            self.encode_data()
            self.split_data()
            self.save_artifacts()

            self.process_anime_data()

            logger.info("Data processing pipeline successful ...")
            
        except CustomException as ce:
            logger.error(str(ce))
            raise CustomException("Error during run ...", sys)
        
if __name__=="__main__":
    data_processor = DataProcessor(ANIMELIST_CSV, PROCESSED_DIR)
    data_processor.run()


