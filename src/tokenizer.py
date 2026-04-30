from spacy.tokens import Token
from spacy.language import Language
import emoji
from argparse import Namespace


def process_emojis(emoji_str: str) -> list[str]:
    ej_list: list[str] = []
    for ej in emoji.emoji_list(emoji_str):
        ej_list.append(ej["emoji"])
    return ej_list


def clean_spaCy_single(text: str, nlp: Language) -> list[Token]:
    doc = nlp(text)
    tokens = [
        token
        for token in doc
        if not any([token.is_punct, token.is_space, token.is_stop])
    ]
    return tokens


def clean_spaCy_batch(
    text: list[str], nlp: Language, batch: int, args: Namespace
) -> list[str]:
    cleaned: list[str] = []

    for doc in nlp.pipe(text, batch_size=batch, disable=["ner", "parser"]):
        if args.verbose:
            print(doc)
        tokens = [
            token
            for token in doc
            if not any([token.is_punct, token.is_space, token.is_stop])
        ]
        cleaned.append(" ".join(token.text for token in tokens))
    return cleaned


def clean_spaCy_non_token(text: str, nlp: Language) -> str:
    doc = nlp(text)
    tokens = [
        token
        for token in doc
        if not any([token.is_punct, token.is_space, token.is_stop])
    ]
    return " ".join([token.text for token in tokens])


def extract_ngram(tokens: list[Token], n: int) -> list[tuple[str, int, int]]:

    n_grams: list[tuple[str, int, int]] = []
    for idx in range(len(tokens) - (n - 1)):
        ngram: list[Token] = tokens[idx : idx + n]
        n_grams.append(
            (
                " ".join([word.text for word in ngram]),
                ngram[0].idx,
                ngram[-1].idx + len(ngram[-1].text),
            )
        )
    return n_grams


def get_emoji_name(e: str) -> str | None:
    data = emoji.EMOJI_DATA.get(e)
    if data is None or "en" not in data:
        return None
    return data["en"].strip(":").replace("_", " ")