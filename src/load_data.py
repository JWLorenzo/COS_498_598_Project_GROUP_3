from huggingface_hub import hf_hub_download  # type: ignore
import pandas as pd
import a_CONSTANTS as C
from pathlib import Path
import numpy as np
import pickle
from pathlib import Path
from numpy.typing import NDArray

THIS_FILES_PATH = Path(__file__)
ROOT_DIR = THIS_FILES_PATH.parents[1]
DATA_PATH = ROOT_DIR / C.DATA_FOLDER_NAME
RENAME_DICT = {"Sentence": "text", "Emoji": "emoji"}
MERGE_NAME = "merged.csv"


def make_data_dir() -> None:
    DATA_PATH.mkdir(exist_ok=True)


def data_exists() -> bool:
    return (DATA_PATH / C.VECTORS).exists() and (DATA_PATH / C.EMOJIS).exists()


def load_or_download(path: Path, repo_id: str, filename: str) -> Path:
    if not path.exists():
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=DATA_PATH,
            )
        )
    return path


def get_dataset_contents() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv_1: Path = load_or_download(
        DATA_PATH / C.TEXT_2_EMOJI_FILE, C.TEXT_2_EMOJI_ID, C.TEXT_2_EMOJI_FILE
    )
    csv_2: Path = load_or_download(
        DATA_PATH / C.SENTIMENT_2_EMOJI_TRAIN,
        C.SENTIMENT_2_EMOJI_ID,
        C.SENTIMENT_2_EMOJI_TRAIN,
    )
    csv_3: Path = load_or_download(
        DATA_PATH / C.SENTIMENT_2_EMOJI_TEST,
        C.SENTIMENT_2_EMOJI_ID,
        C.SENTIMENT_2_EMOJI_TEST,
    )
    return pd.read_csv(csv_1), pd.read_csv(csv_2), pd.read_csv(csv_3)


def process_dataframes(
    df_1: pd.DataFrame, df_2: pd.DataFrame, df_3: pd.DataFrame
) -> pd.DataFrame:
    df_2.columns = df_2.columns.str.strip()
    df_3.columns = df_3.columns.str.strip()
    df_2 = df_2.rename(columns=RENAME_DICT)
    df_3 = df_3.rename(columns=RENAME_DICT)
    df_4 = pd.concat([df_1, df_2, df_3], ignore_index=True)
    df_4.to_csv(DATA_PATH / MERGE_NAME)

    return df_4


def save_data(vec_list: NDArray[np.float32], emojis: list[str]) -> None:

    np.save(DATA_PATH / C.VECTORS, vec_list)
    with open(DATA_PATH / C.EMOJIS, "wb") as f:
        pickle.dump(emojis, f)


def load_data() -> tuple[NDArray[np.float32], list[str]]:
    with open(DATA_PATH / C.EMOJIS, "rb") as f:
        data = pickle.load(f)

    return (np.load(DATA_PATH / C.VECTORS), data)
