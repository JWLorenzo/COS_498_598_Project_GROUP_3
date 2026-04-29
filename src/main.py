import a_CONSTANTS as C
import tokenizer as T
import load_data as D
import pandas as pd
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
import numpy as np
import spacy
from spacy.language import Language

SAMPLE = 20000
BATCH_SIZE = 128

USE_SAMPLE = False
USE_GPU = True
RERUN = False
MAX_N = 2
REPLACEMENTS = 6
BATCH = 64
CLEAN = True


def create_emoji_mapping(
    merged: pd.DataFrame, model: SentenceTransformer, nlp: Language
) -> list[tuple[str, NDArray[np.float32]]]:
    if CLEAN:
        sents: list[str] = T.clean_spaCy_batch(merged["text"].tolist(), nlp, BATCH)
    else:
        sents: list[str] = merged["text"].tolist()
    emoji_set: set[str] = set(
        emoji for emojis in merged["emoji_list"].tolist() for emoji in emojis
    )
    vectors: NDArray[np.float32] = encoder(model, sents)

    # This list comprehension is just creating a mapping of emojis to vectors where we add the vector to that associated emojis list if that emoji is in the translated string
    emoji_mapping: list[tuple[str, list[NDArray[np.float32]]]] = [
        (
            emoji,
            [
                vector
                for vector, emojis in zip(vectors, merged["emoji_list"].to_list())
                if emoji in emojis
            ],
        )
        for emoji in emoji_set
    ]

    # We just get all of the emoji - vector pairs, but average the vector
    return [(emoji, np.mean(vec, 0)) for emoji, vec in emoji_mapping]


def encoder(
    model: SentenceTransformer, t_input: list[str] | str | pd.Series
) -> NDArray[np.float32]:
    # Normalized the vectors to make the cosine similarity easier
    return np.array(
        model.encode(  # type: ignore
            t_input,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    )


def initialize_data(
    model: SentenceTransformer, nlp: Language
) -> tuple[list[str], NDArray[np.float32]]:
    merged: pd.DataFrame = D.process_dataframes(*D.get_dataset_contents())
    merged = merged.dropna(subset=["text"])

    if USE_SAMPLE:
        merged = merged[:SAMPLE]

    merged["emoji_list"] = merged["emoji"].apply(T.process_emojis)
    mapping: list[tuple[str, NDArray[np.float32]]] = create_emoji_mapping(
        merged, model, nlp
    )
    vectors: NDArray[np.float32] = np.array([vector for _, vector in mapping])
    emojis: list[str] = [emoji for emoji, _ in mapping]
    return (emojis, vectors)


def get_similarities(
    vector: NDArray[np.float32], vectors: NDArray[np.float32]
) -> NDArray[np.float32]:

    return vectors @ vector


def get_top_k(similarities: NDArray[np.float32], k: int) -> NDArray[np.intp]:
    indices: NDArray[np.intp] = np.argpartition(similarities, -k)[-k:]
    return indices[np.argsort(similarities[indices])[::-1]]


def get_emoji_slices(
    selection_sorted: list[tuple[tuple[str, int, int], str, np.float32]],
) -> list[tuple[tuple[str, int, int], str, np.float32]]:
    selected: list[tuple[tuple[str, int, int], str, np.float32]] = []
    count: int = 0
    curr_idx: int = 0
    running: bool = True
    while running:
        enabled: bool = True
        for ngram in selected:
            s1: int = selection_sorted[curr_idx][0][1]
            e1: int = selection_sorted[curr_idx][0][2]
            s2: int = ngram[0][1]
            e2: int = ngram[0][2]
            if not (s1 >= e2 or s2 >= e1):
                enabled = False
        if enabled:
            selected.append(selection_sorted[curr_idx])
            count += 1
        curr_idx += 1

        if count >= REPLACEMENTS or curr_idx > len(selection_sorted) - 1:
            running = False
    return sorted(selected, key=lambda x: float(x[0][1]))


def construct_sentence(
    selected_sorted: list[tuple[tuple[str, int, int], str, np.float32]], word_input: str
) -> str:
    sliced_index: int = 0
    final_list: str = ""
    for ngram in selected_sorted:
        final_list += word_input[sliced_index : ngram[0][1]] + ngram[1]
        sliced_index = ngram[0][2]
    final_list += word_input[sliced_index:]
    return final_list


def main(model: SentenceTransformer, nlp: Language) -> None:
    if not D.make_data_dir() or RERUN:
        emojis, vec_array = initialize_data(model, nlp)
        D.save_data(vec_array, emojis)
    else:
        vec_array, emojis = D.load_data()

    word_input: str = (
        "Being a nurse is a rollercoaster of emotions, from comforting patients to dealing with medical emergencies."
    )
    n_grams: list[tuple[str, int, int]] = [
        n_gram
        for n in range(1, MAX_N + 1)
        for n_gram in T.extract_ngram(T.clean_spaCy_single(word_input, nlp), n)
    ]

    similarities: NDArray[np.float32] = np.array(
        [get_similarities(encoder(model, n_gram[0]), vec_array) for n_gram in n_grams]
    )

    similarity_struct: list[tuple[tuple[str, int, int], str, np.float32]] = [
        (ngram, emojis[idx := get_top_k(similarity, 1)[-1]], similarity[idx])
        for ngram, similarity in zip(n_grams, similarities)
    ]

    selection_sorted: list[tuple[tuple[str, int, int], str, np.float32]] = sorted(
        similarity_struct, reverse=True, key=lambda x: float(x[2])
    )

    selected_sorted: list[tuple[tuple[str, int, int], str, np.float32]] = (
        get_emoji_slices(selection_sorted)
    )

    final_list: str = construct_sentence(selected_sorted, word_input)
    print(selected_sorted)
    print(final_list)


if __name__ == "__main__":
    nlp: Language = spacy.load("en_core_web_sm")
    # https://sbert.net/ for model
    if USE_GPU:
        model = SentenceTransformer(C.MODEL, device="cuda")
    else:
        model = SentenceTransformer(C.MODEL)
    main(model, nlp)
