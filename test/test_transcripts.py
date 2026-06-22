import random
from Bio import SeqIO

def extract_random_fasta(input_fasta, output_fasta, num_sequences=1000):
    """
    Randomly extract a specified number of sequences from the input FASTA file.
    """
    print(f"Reading file: {input_fasta} ...")
    
    # Parse the FASTA file and load all records into a list.
    # Note: For extremely large FASTA files (e.g., tens of GBs), this approach consumes significant memory.
    # It works perfectly for standard transcriptomes (e.g., human transcriptome ~200MB).
    records = list(SeqIO.parse(input_fasta, "fasta"))
    total_seqs = len(records)
    
    print(f"Found a total of {total_seqs} sequences.")
    
    # Check if the requested number exceeds the actual number of sequences in the file
    if total_seqs < num_sequences:
        print(f"Warning: Input file only contains {total_seqs} sequences. Extracting all available sequences.")
        num_sequences = total_seqs
        
    # Random sampling without replacement
    print(f"Randomly extracting {num_sequences} sequences...")
    sampled_records = random.sample(records, num_sequences)
    
    # Write the sampled sequences to a new FASTA file
    SeqIO.write(sampled_records, output_fasta, "fasta")
    print(f"✅ Extraction complete! Test dataset saved to: {output_fasta}")


if __name__ == "__main__":
    INPUT_FILE = "./gencode.v43.pc_transcripts.fa" 
    OUTPUT_FILE = "./gencode.v43.pc_transcripts.test_2000.fa"
    
    extract_random_fasta(INPUT_FILE, OUTPUT_FILE, num_sequences=2000)