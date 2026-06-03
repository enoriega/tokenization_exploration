"""
Save a HuggingFace tokenizers JSON file as a directory loadable by AutoTokenizer.from_pretrained().
"""
import argparse
from pathlib import Path
from tokenizers import Tokenizer, decoders
from transformers import PreTrainedTokenizerFast


def save_for_autotokenizer(tokenizer_json: Path, output_dir: Path) -> None:
    backend = Tokenizer.from_file(str(tokenizer_json))

    # If the tokenizer uses ByteLevel pre-tokenization but has no decoder, inject one
    # so that AutoTokenizer.decode() reverses the Ġ-space encoding correctly.
    pre_tok = backend.pre_tokenizer
    if backend.decoder is None and pre_tok is not None and "ByteLevel" in type(pre_tok).__name__:
        backend.decoder = decoders.ByteLevel()
        print("Note: added missing ByteLevel decoder (source tokenizer had none)")

    # Detect special tokens defined in the tokenizer JSON; decoder is {id: AddedToken}
    added = {tok.content: tok_id for tok_id, tok in backend.get_added_tokens_decoder().items() if tok.special}

    eos = next((c for c in ("<|endoftext|>",) if c in added), None)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token=eos,
        unk_token=eos,  # BPE has no true UNK; reuse EOS as fallback
        pad_token=eos,
        model_max_length=backend.truncation["max_length"] if backend.truncation else int(1e30),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved tokenizer to {output_dir}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    if eos:
        print(f"EOS / PAD / UNK token: '{eos}' (id {added[eos]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a tokenizers JSON file to an AutoTokenizer-compatible directory."
    )
    parser.add_argument("tokenizer_json", type=Path, help="Path to the tokenizer JSON file")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Output directory (default: <tokenizer_json stem>_pretrained/)",
    )
    args = parser.parse_args()

    out = args.output_dir or args.tokenizer_json.parent / (args.tokenizer_json.stem + "_pretrained")
    save_for_autotokenizer(args.tokenizer_json, out)
