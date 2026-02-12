import os, json, time, re
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# ----------------- Settings -----------------
OUTDIR      = "fw_edu_tokens"                 # output folder
DATASET     = "HuggingFaceFW/fineweb-edu"     # dataset id
NAME        = "sample-350BT"                   # None (huge), "sample-10BT", "sample-100BT", "sample-350BT", or "CC-MAIN-YYYY-WW"
SPLIT       = "train"
TOKENIZER   = "gpt2"
ADD_EOS     = True
CHECKPOINT_TOKS = 100_000_000                 # checkpoint every 100M tokens
MIN_INT_SCORE   = None                        # e.g., 80 to filter lower-quality docs
# retry/skip knobs (short + strict as requested)
RETRY_LIMIT_PER_SHARD = 3                     # max retries per shard on 429
RETRY_BACKOFFS = [0.5, 1.0, 2.0]              # seconds

# Optional: tame burstiness a bit
os.environ.setdefault("HF_HUB_MAX_RETRIES", "0")
os.environ.setdefault("HF_HUB_TIMEOUT", "60")

# ------------- Checkpoint helpers -----------
def load_checkpoint():
    ckpt_path = os.path.join(OUTDIR, "checkpoint.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    # start fresh
    return {"n_tokens": 0, "doc_idx": 0}

def save_checkpoint(n_tokens, doc_idx):
    ckpt_path = os.path.join(OUTDIR, "checkpoint.json")
    with open(ckpt_path, "w") as f:
        json.dump({"n_tokens": int(n_tokens), "doc_idx": int(doc_idx)}, f)

# ----------------- Loader -------------------
def load_fw_stream():
    """Load FineWeb-Edu streaming dataset with optional token & named config."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    # A couple quick retries in case repo-info is transiently throttled
    for delay in [0.5, 1.0, 2.0]:
        try:
            return load_dataset(DATASET, name=NAME, split=SPLIT, streaming=True, token=token)
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️  429 during dataset open. Sleeping {delay}s and retrying…")
                time.sleep(delay)
                continue
            else:
                raise
    # last attempt (let it raise if still failing)
    return load_dataset(DATASET, name=NAME, split=SPLIT, streaming=True, token=token)

# ------------- Main -------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    tok.model_max_length = int(1e9)  # silence warnings
    eos_id = tok.eos_token_id

    # Load/resume checkpoint
    ckpt = load_checkpoint()
    n_tokens = int(ckpt["n_tokens"])
    doc_idx_start = int(ckpt["doc_idx"])
    print(f"▶ Resuming from doc {doc_idx_start:,}, token {n_tokens:,}")

    # Prepare backing memmap (uint16)
    cap = max(n_tokens + 1_000_000, 512 * 1024 * 1024)
    path_bin = os.path.join(OUTDIR, "tokens.bin")
    if not os.path.exists(path_bin):
        with open(path_bin, "wb") as f:
            f.truncate(cap * 2)
    fp = np.memmap(path_bin, mode="r+", dtype=np.uint16, shape=(cap,))

    def ensure(extra):
        nonlocal fp, cap, n_tokens
        need = n_tokens + extra
        if need <= cap:
            return
        new_cap = cap
        while new_cap < need:
            new_cap = int(new_cap * 1.7)
        fp.flush()
        del fp
        with open(path_bin, "ab") as f:
            f.truncate(new_cap * 2)
        fp = np.memmap(path_bin, mode="r+", dtype=np.uint16, shape=(new_cap,))
        cap = new_cap

    # Stream FineWeb-Edu
    ds = load_fw_stream()

    # ---- Robust iteration with retry/skip on 429s ---------------------------
    from huggingface_hub.errors import HfHubHTTPError
    PARQUET_RE = re.compile(r"(train-\d{5}-of-\d{5}\.parquet)")
    per_shard_tries = {}

    def extract_shard_name(exc_msg: str):
        m = PARQUET_RE.search(exc_msg or "")
        return m.group(1) if m else "unknown-shard"

    def new_iter(d):
        return iter(d)

    dset = ds
    it = new_iter(dset)
    doc_idx = -1

    while True:
        try:
            ex = next(it)
            doc_idx += 1
        except StopIteration:
            break
        except Exception as e:
            msg = str(e)
            if ("429" not in msg) and (not isinstance(e, HfHubHTTPError)):
                raise

            shard = extract_shard_name(msg)
            tries = per_shard_tries.get(shard, 0)

            if tries < RETRY_LIMIT_PER_SHARD:
                delay = RETRY_BACKOFFS[min(tries, len(RETRY_BACKOFFS)-1)]
                per_shard_tries[shard] = tries + 1
                print(f"⚠️  429 on {shard} (retry {tries+1}/{RETRY_LIMIT_PER_SHARD}) — sleeping {delay}s…")
                time.sleep(delay)
                # rebuild iterator and fast-forward past already processed docs
                try:
                    # try reshuffling if supported in your datasets version
                    dset = dset.shuffle(seed=(int(time.time()) & 0xFFFF))
                except AttributeError:
                    # no shuffle available; proceed without reshuffle
                    pass
                dset = dset.skip(max(doc_idx, 0))  # skip what we've already processed
                it = new_iter(dset)
                continue
            else:
                print(f"⏭️  Skipping hot shard {shard}; moving on.")
                per_shard_tries[shard] = 0
                try:
                    dset = dset.shuffle(seed=(int(time.time()) & 0xFFFF))
                except AttributeError:
                    pass
                dset = dset.skip(max(doc_idx, 0))
                it = new_iter(dset)
                continue

        # -------- your normal processing below (only runs after a successful read) -----
        if doc_idx < doc_idx_start:
            continue

        # Optional quality filter
        if MIN_INT_SCORE is not None:
            try:
                if int(ex.get("int_score", 0)) < int(MIN_INT_SCORE):
                    continue
            except Exception:
                pass

        text = ex.get("text") or ""
        if not text:
            continue

        ids = tok(text, add_special_tokens=False)["input_ids"]
        if ADD_EOS and eos_id is not None:
            ids.append(eos_id)

        arr = np.asarray(ids, dtype=np.int64)
        if arr.size == 0:
            continue
        if arr.max(initial=0) >= 65536:
            raise ValueError("Token id exceeds uint16 capacity; use a different dtype or tokenizer.")

        ensure(arr.size)
        fp[n_tokens:n_tokens + arr.size] = arr.astype(np.uint16, copy=False)
        n_tokens += arr.size

        # periodic checkpoint
        if n_tokens // CHECKPOINT_TOKS != (n_tokens - arr.size) // CHECKPOINT_TOKS:
            fp.flush()
            save_checkpoint(n_tokens, doc_idx + 1)
            print(f"💾 Checkpoint: {n_tokens:,} tokens, doc {doc_idx+1:,}")

    # Finalize
    fp.flush()
    meta = {
        "tokenizer": TOKENIZER,
        "dtype": "uint16",
        "num_tokens": int(n_tokens),
        "add_eos_per_doc": ADD_EOS,
        "dataset": DATASET,
        "name": NAME or "default",
        "split": SPLIT,
        "min_int_score": MIN_INT_SCORE,
        "note": "Flat token stream; resumable; short 429 retries + shard skipping.",
    }
    with open(os.path.join(OUTDIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    save_checkpoint(n_tokens, doc_idx + 1)
    print(f"✅ Finished. Wrote {n_tokens:,} tokens (~{n_tokens*2/1e9:.2f} GB).")

if __name__ == "__main__":
    main()
