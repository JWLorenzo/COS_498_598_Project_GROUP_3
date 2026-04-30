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
from spacy.tokens import Token


def create_emoji_mapping(
    merged: pd.DataFrame, model: SentenceTransformer, nlp: Language, args: Namespace
) -> list[tuple[str, NDArray[np.float32]]]:
    if not args.emoji:
        sents: list[str] = T.clean_spaCy_batch(
            merged["text"].tolist(), nlp, args.batch, args
        )
    else:
        sents = merged["text"].tolist()
    emoji_set: set[str] = set(
        emoji for emojis in merged["emoji_list"].tolist() for emoji in emojis
    )
    vectors: NDArray[np.float32] = encoder(model, sents, args)

    # This list comprehension is just creating a mapping of emojis to vectors
    # where we add the vector to that associated emojis list if that emoji is in the translated string

    emoji_mapping: list[tuple[str, list[NDArray[np.float32]]]] = []
    for em in emoji_set:
        matched_vectors: list[NDArray[np.float32]] = [
            vector
            for vector, emojis in zip(vectors, merged["emoji_list"].tolist())
            if em in emojis
        ]
        emoji_mapping.append((em, matched_vectors))

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
        merged = pd.DataFrame.from_dict(
            {
                "emoji_list": [[ej] for ej in emoji.EMOJI_DATA],
                "text": [
                    text["en"].strip(":").replace("_", " ")
                    for text in emoji.EMOJI_DATA.values()
                ],
            }
        )
        mapping = create_emoji_mapping(merged, model, nlp, args)

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


def overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return not (e1 <= s2 or e2 <= s1)


def duplicate(em1: str, em2: str) -> bool:
    return em1 == em2


def get_emoji_slices(
    selection_sorted: list[tuple[tuple[str, int, int], str, np.float32]],
    args: Namespace,
) -> list[tuple[tuple[str, int, int], str, np.float32]]:
    selected: list[tuple[tuple[str, int, int], str, np.float32]] = []
    count: int = 0
    curr_idx: int = 0
    while count < args.repl or curr_idx < len(selection_sorted):
        s1: int = selection_sorted[curr_idx][0][1]
        e1: int = selection_sorted[curr_idx][0][2]
        no_overlap: bool = not any(
            overlap(s1, e1, ngram[0][1], ngram[0][2]) for ngram in selected
        )
        no_duplicate: bool = not any(
            duplicate(selection_sorted[curr_idx][1], ngram[1]) for ngram in selected
        )
        if no_overlap and no_duplicate:
            selected.append(selection_sorted[curr_idx])
            count += 1
        curr_idx += 1
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


def run_translation(
    model: SentenceTransformer,
    nlp: Language,
    vec_array: NDArray[np.float32],
    emojis: list[str],
    args: Namespace,
) -> None:
    tokens: list[Token] = T.clean_spaCy_single(args.text, nlp)

    n_grams: list[tuple[str, int, int]] = [
        n_gram
        for n in range(1, args.ngram + 1)
        for n_gram in T.extract_ngram(tokens, n)
    ]

    similarities: NDArray[np.float32] = np.array(
        [
            get_similarities(n_vec, vec_array)
            for n_vec in encoder(model, [ngram[0] for ngram in n_grams], args)
        ]
    )

    similarity_struct: list[tuple[tuple[str, int, int], str, np.float32]] = []
    for ngram, similarity in zip(n_grams, similarities):
        idx: np.intp = get_top_k(similarity, 1)[-1]
        similarity_struct.append((ngram, emojis[idx], similarity[idx]))

    selection_sorted: list[tuple[tuple[str, int, int], str, np.float32]] = sorted(
        similarity_struct, reverse=True, key=lambda x: float(x[2])
    )

    selected_sorted: list[tuple[tuple[str, int, int], str, np.float32]] = (
        get_emoji_slices(selection_sorted, args)
    )

    final_list: str = construct_sentence(selected_sorted, args.text)
    print(selected_sorted)
    print(final_list)


def check_positive(value: str):
    try:
        val: int = int(value)
        if val <= 0:
            raise ArgumentTypeError(f"{val} must be a positive integer greater than 0")
    except ValueError:
        raise Exception(f"{value} is not an integer")
    return val


def build_parser() -> ArgumentParser:
    parser: ArgumentParser = ArgumentParser(
        description="Minimizing Language With Emojis"
    )
    parser.add_argument(
        "-e", "--emoji", help="Use emoji dict instead of corpus", action="store_true"
    )

    parser.add_argument("-v", "--verbose", help="Verbosity", action="store_true")

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
    return parser


def parse_args() -> Namespace:
    return build_parser().parse_args()


def main(model: SentenceTransformer, nlp: Language, args: Namespace) -> None:
    D.make_data_dir()
    if not D.data_exists() or args.rerun:
        emojis, vec_array = initialize_data(model, nlp, args)
        D.save_data(vec_array, emojis)
    else:
        vec_array, emojis = D.load_data()

    if not args.rerun:
        run_translation(model, nlp, vec_array, emojis, args)

    else:
        print("Vectorization complete")


if __name__ == "__main__":
    nlp: Language = spacy.load("en_core_web_sm")

    args: Namespace = parse_args()

    if args.gpu:
        model = SentenceTransformer(C.MODEL, device="cuda")
    else:
        model = SentenceTransformer(C.MODEL)
    main(model, nlp, args)
