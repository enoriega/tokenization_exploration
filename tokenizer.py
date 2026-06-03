from pubmed import build_dataset
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

def train_pubmed_tokenizer(data_dir:str | Path):
    dataset = build_dataset(*Path(data_dir).absolute().glob("*.gz"))
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=50368, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator(get_training_corpus(dataset), trainer=trainer)
    return tokenizer

def get_training_corpus(dataset, batch_size=1000, text_column="text"):
    """Generator function that yields batches of texts"""
    for start_idx in range(0, len(dataset), batch_size):
        batch = dataset[start_idx : start_idx + batch_size]
        yield batch[text_column]


if __name__ == "__main__":
    tokenizer = train_pubmed_tokenizer("/Users/enoriega/github/tokenization_exploration/data/ftp.ncbi.nlm.nih.gov/pubmed/baseline")
    encoding = tokenizer.encode("""OBJECTIVE: Iodide transport defect (ITD) is a rare disorder characterised by an inability of the thyroid to maintain an iodide gradient across the basolateral membrane of thyroid follicular cells, that often results in congenital hypothyroidism. When present the defect is also found in the salivary glands and gastric mucosa and it has been shown to arise from abnormalities of the sodium/iodide symporter (NIS). PATIENT: We describe a woman with hypothyroidism identified at the 3rd month of life. The diagnosis of ITD was suspected because of nodular goitre, and little if any iodide uptake by the thyroid and salivary glands. Treatment with iodide partially corrected the hypothyroidism; however, long-term substitution therapy with L-thyroxine was started. MEASUREMENTS: Thyroid radioiodide uptake was only 1.4% and 0.3% at 1 and 24 h after the administration of recombinant human TSH. The saliva to plasma I- ratio was 1.1 indicating that the inability of the thyroid gland to concentrate I- was also present in the salivary glands. RESULTS: Analysis of the patient's NIS gene revealed a 15 nucleotide (nt) deletion of the coding sequence (nt 1314 through nt 1328) and the insertion of 15 nt duplicating the first 15 nt of the adjacent intron. The patient was homozygous for this insertion/deletion, while both consanguineous parents were heterozygous. This deletion predicts the production of a protein lacking the five terminal amino acids of exon XI (439-443) which are located in the 6th intracellular loop. COS-7 cells transfected with a vector expressing the mutant del-(439-443) NIS failed to concentrate iodide, suggesting that the mutation was the direct cause of the ITD in this patient. CONCLUSION: In conclusion we describe the first Italian case of congenital hypothyroidism due to a new deletion in the NIS gene.""")

    tokenizer.save("data/tokenizer_test")
    tokens = encoding.tokens
    tokens_str = " ".join(tokens)
    print(tokens_str)
