# COS_498_NLP_Project - Group 3
This project augments natural-language sentences with semantically appropriate emojis using sentence-transformer embeddings and per-emoji centroid vectors

# 1. Contents
/chunk_aligned_cache
- Contains pre encoded vectors and pickled emojis

/src
- a_CONSTANTS.py
    - Contains our constants for our program

- load_data.py
    - Contains the functions related to handling (loading, processing, etc)

- main.py
    - It's the heart of the project, this is the script you'll call to see our work. See section 7 for examples.

- tokenizer.py
    - Contains the functions related to tokenizing sentences, cleaning stopwords, spaces, etc. 

/z_data
- Directory generated after running the main.py script.

# 2. Make a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 3.1 GPU acceleration 
If you have a NVIDIA GPU: 
Go to: https://pytorch.org/get-started/locally/
and download a CUDA-enabled PyTorch build that matches your system. 

Use flag -g to enable GPU use. 

# 4. Download the spaCy English model
python -m spacy download en_core_web_sm

# 5. Run the pipeline
python src/main.py [arguments]

On the first run, the datasets are downloaded from Hugging Face into z_data/,
the merged corpus is built, and the emoji centroids are computed and cached as
z_data/vectors.npy and z_data/emojis.pkl. Subsequent runs reuse the cache
unless -r / --rerun is passed.

# 6. Arguments
Each argument has a short and long form (e.g. `-t` or `--text`).
 
**`-t`, `--text`** *string, default: "The quick brown fox jumps over the lazy dog"*
Input sentence to translate into emoji-augmented form.
 
**`-e`, `--emoji`** *flag, default: off*
Build centroids from the unicode emoji dictionary instead of from the corpus.
Each emoji's centroid is the embedding of its unicode short name
(e.g. 🎢 → *"roller coaster"*). No corpus contribution.
 
**`-P`, `--phrase`** *flag, default: off*
Use phrase-aligned centroids instead of the default whole-sentence averaging.
Each emoji is matched to the n-gram chunk of each row whose embedding is
closest to the emoji's unicode-name vector, with neighbor smoothing.
Encoding takes considerably longer than other options and is memory intensive. 
Ignored if `--emoji` is also set.
 
**`-s`, `--smooth`** *float, default: 0.7*
Phrase-mode only. Weight on the best-matching chunk vs. its immediate
neighbors when building each contribution. `1.0` = best chunk only;
`0.0` = neighbors only.
 
**`-g`, `--gpu`** *flag, default: off*
Run the sentence-transformer on CUDA. Requires an NVIDIA GPU and a
CUDA-enabled PyTorch build (see 2.1).
 
**`-r`, `--rerun`** *flag, default: off*
Force the centroid cache to rebuild even if `vectors.npy` and `emojis.pkl`
already exist. Required when switching between `--phrase` and the default
whole-sentence method, or when changing the corpus.
 
**`-n`, `--ngram`** *positive int, default: 2*
Maximum n-gram size to consider during inference (and during phrase alignment
when `--phrase` is set).
 
**`-p`, `--repl`** *positive int, default: 4*
Maximum number of non-overlapping emoji insertions in the output sentence.
 
**`-b`, `--batch`** *positive int, default: 64*
Batch size for the sentence encoder. Larger values are faster on GPU but use
more memory.

# 7. Examples
# Default whole-sentence method on a custom sentence
python src/main.py -t "Rain or shine we play outside all day long"

# Phrase-aligned method with GPU
python src/main.py -P -g -t "Rain or shine we play outside all day long"

# Switch from whole-sentence to phrase-aligned (must rerun to rebuild cache)
python src/main.py -P -g -r -t "Rain or shine we play outside all day long"

# Unicode-name-only baseline
python src/main.py -e -t "Rain or shine we play outside all day long"

# Larger n-grams, more replacements
python src/main.py -n 3 -p 6 -t "Rain or shine we play outside all day long"