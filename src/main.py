import a_CONSTANTS as C
import tokenizer as T
import load_data as D
import pandas as pd
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
import numpy as np
import spacy
from spacy.language import Language
from argparse import ArgumentParser
from argparse import Namespace
from argparse import ArgumentTypeError
import emoji


def create_emoji_mapping(
    merged: pd.DataFrame, model: SentenceTransformer, nlp: Language, args: Namespace
) -> list[tuple[str, NDArray[np.float32]]]:
    sents: list[str] = T.clean_spaCy_batch(merged["text"].tolist(), nlp, args.batch)
    emoji_set: set[str] = set(
        emoji for emojis in merged["emoji_list"].tolist() for emoji in emojis
    )
    vectors: NDArray[np.float32] = encoder(model, sents, args)

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
    model: SentenceTransformer, t_input: list[str] | str | pd.Series, args: Namespace
) -> NDArray[np.float32]:
    # Normalized the vectors to make the cosine similarity easier
    return np.array(
        model.encode(  # type: ignore
            t_input,
            batch_size=args.batch,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    )


def initialize_data(
    model: SentenceTransformer, nlp: Language, args: Namespace
) -> tuple[list[str], NDArray[np.float32]]:
    if not args.emoji:
        merged: pd.DataFrame = D.process_dataframes(*D.get_dataset_contents())
        merged = merged.dropna(subset=["text", "emoji"])

        merged["emoji_list"] = merged["emoji"].apply(T.process_emojis)
        mapping: list[tuple[str, NDArray[np.float32]]] = create_emoji_mapping(
            merged, model, nlp, args
        )
    else:
        merged: pd.DataFrame = pd.DataFrame.from_dict(
            {
                "emoji_list": [ej for ej in emoji.EMOJI_DATA],
                "text": [
                    text["en"].strip(":").replace("_", " ")
                    for text in emoji.EMOJI_DATA.values()
                ],
            }
        )
        mapping: list[tuple[str, NDArray[np.float32]]] = create_emoji_mapping(
            merged, model, nlp, args
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
    args: Namespace,
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

        if count >= args.repl or curr_idx > len(selection_sorted) - 1:
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


def main(model: SentenceTransformer, nlp: Language, args: Namespace) -> None:
    if not D.make_data_dir() or args.rerun:
        emojis, vec_array = initialize_data(model, nlp, args)
        D.save_data(vec_array, emojis)
    else:
        vec_array, emojis = D.load_data()

    n_grams: list[tuple[str, int, int]] = [
        n_gram
        for n in range(1, args.ngram + 1)
        for n_gram in T.extract_ngram(T.clean_spaCy_single(args.text, nlp), n)
    ]

    similarities: NDArray[np.float32] = np.array(
        [
            get_similarities(n_vec, vec_array)
            for n_vec in encoder(model, [ngram[0] for ngram in n_grams], args)
        ]
    )

    similarity_struct: list[tuple[tuple[str, int, int], str, np.float32]] = [
        (ngram, emojis[idx := get_top_k(similarity, 1)[-1]], similarity[idx])
        for ngram, similarity in zip(n_grams, similarities)
    ]

    selection_sorted: list[tuple[tuple[str, int, int], str, np.float32]] = sorted(
        similarity_struct, reverse=True, key=lambda x: float(x[2])
    )

    selected_sorted: list[tuple[tuple[str, int, int], str, np.float32]] = (
        get_emoji_slices(selection_sorted, args)
    )

    final_list: str = construct_sentence(selected_sorted, args.text)
    print(selected_sorted)
    print(final_list)


def check_positive(value: int):
    try:
        value = int(value)
        if value <= 0:
            raise ArgumentTypeError(
                f"{value} must be a positive integer greater than 0"
            )
    except ValueError:
        raise Exception(f"{value} is not an integer")
    return value


if __name__ == "__main__":
    nlp: Language = spacy.load("en_core_web_sm")

    parser: ArgumentParser = ArgumentParser(
        description="Minimizing Language With Emojis"
    )
    parser.add_argument(
        "-e", "--emoji", help="Use emoji dict instead of corpus", action="store_true"
    )

    parser.add_argument("-g", "--gpu", help="Use CUDA?", action="store_true")
    parser.add_argument("-r", "--rerun", help="Rerun the pipeline", action="store_true")
    parser.add_argument(
        "-n", "--ngram", help="Max ngram size", type=check_positive, default=2
    )
    parser.add_argument(
        "-p", "--repl", help="Num ngrams to replace", type=check_positive, default=4
    )
    parser.add_argument(
        "-b", "--batch", help="Batch size for encoding", type=check_positive, default=64
    )

    parser.add_argument(
        "-t",
        "--text",
        help="Enter a string to translate",
        type=str,
        default="The quick brown fox jumps over the lazy dog",
    )

    args: Namespace = parser.parse_args()

    if args.gpu:
        model = SentenceTransformer(C.MODEL, device="cuda")
    else:
        model = SentenceTransformer(C.MODEL)
    main(model, nlp, args)
