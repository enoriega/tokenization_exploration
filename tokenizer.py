from pathlib import Path
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from transformers import PreTrainedTokenizerFast

def train_pubmed_tokenizer(dataset):
    # Special tokens: BERT-style set for ModernBERT compatibility.
    # [PAD]=0 by convention. Reserved slots let you add task tokens later
    # without resizing the embedding matrix.
    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    # Pad out to a multiple of 64. 5 specials + 59 reserved = 64 total specials,
    # vocab_size 50368 stays a multiple of 64.
    RESERVED = [f"[unused{i}]" for i in range(59)]
    ALL_SPECIALS = SPECIAL_TOKENS + RESERVED

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
     # add_prefix_space=True so the first word tokenizes like a mid-sequence word
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=50368,
        special_tokens=ALL_SPECIALS,
        min_frequency=3,
        # Guarantee the full 256-byte alphabet so rare bytes never UNK
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(dataset, trainer=trainer)

    # BERT-style templating post-processor
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    transformers_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        # These map roles to the actual token strings so AutoTokenizer
        # knows which is which, independent of vocab position.
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=8192,   # ModernBERT native context
    )
    

    return tokenizer, transformers_tokenizer

def get_training_corpus(dataset, batch_size=1000, text_column="text"):
    """Generator function that yields batches of texts"""
    for start_idx in range(0, len(dataset), batch_size):
        batch = dataset[start_idx : start_idx + batch_size]
        yield batch[text_column]


if __name__ == "__main__":
    tokenizer, pretrained_tokenizer = train_pubmed_tokenizer("/Users/enoriega/github/tokenization_exploration/data/ftp.ncbi.nlm.nih.gov/pubmed/baseline")
    encoding = tokenizer.encode("""OBJECTIVE: Iodide transport defect (ITD) is a rare disorder characterised by an inability of the thyroid to maintain an iodide gradient across the basolateral membrane of thyroid follicular cells, that often results in congenital hypothyroidism. When present the defect is also found in the salivary glands and gastric mucosa and it has been shown to arise from abnormalities of the sodium/iodide symporter (NIS). PATIENT: We describe a woman with hypothyroidism identified at the 3rd month of life. The diagnosis of ITD was suspected because of nodular goitre, and little if any iodide uptake by the thyroid and salivary glands. Treatment with iodide partially corrected the hypothyroidism; however, long-term substitution therapy with L-thyroxine was started. MEASUREMENTS: Thyroid radioiodide uptake was only 1.4% and 0.3% at 1 and 24 h after the administration of recombinant human TSH. The saliva to plasma I- ratio was 1.1 indicating that the inability of the thyroid gland to concentrate I- was also present in the salivary glands. RESULTS: Analysis of the patient's NIS gene revealed a 15 nucleotide (nt) deletion of the coding sequence (nt 1314 through nt 1328) and the insertion of 15 nt duplicating the first 15 nt of the adjacent intron. The patient was homozygous for this insertion/deletion, while both consanguineous parents were heterozygous. This deletion predicts the production of a protein lacking the five terminal amino acids of exon XI (439-443) which are located in the 6th intracellular loop. COS-7 cells transfected with a vector expressing the mutant del-(439-443) NIS failed to concentrate iodide, suggesting that the mutation was the direct cause of the ITD in this patient. CONCLUSION: In conclusion we describe the first Italian case of congenital hypothyroidism due to a new deletion in the NIS gene.""")

    tokenizer.save("data/tokenizer_test")
    tokens = encoding.tokens
    tokens_str = " ".join(tokens)
    print(tokens_str)
