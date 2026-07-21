import pickle
from itertools import groupby


__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"


# load fasta files
def fasta_iter(fasta_file):
    """
    given a fasta file, yield tuples of header, sequence
    """
    with open(fasta_file) as file:
        # ditch the boolean (x[0]) and just keep the header or sequence since
        faiter = (x[1] for x in groupby(file, lambda line: line[0] == ">"))
        for header in faiter:
            header = header.__next__()[1:].strip() # drop the ">"
            seq = "".join(s.strip() for s in faiter.__next__()) # join all sequences
            yield header, seq

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Build transcript sequence pickle from FASTA + metadata")
    parser.add_argument("--tx_meta", required=True, help="Path to transcript_meta.pkl")
    parser.add_argument("--fasta", required=True, help="Path to transcript FASTA file")
    parser.add_argument("--output", required=True, help="Output pickle path")
    parser.add_argument("--id_sep", default="|", help="FASTA header ID separator (default: '|')")
    parser.add_argument("--id_field", type=int, default=0,
                        help="Which field after split to use as transcript ID (default: 0)")
    args = parser.parse_args()

    import pickle
    with open(args.tx_meta, "rb") as f:
        tx_meta = pickle.load(f)

    tx_seq = {}
    for h, s in fasta_iter(args.fasta):
        tx_id = h.split(args.id_sep)[args.id_field]
        if tx_id in tx_meta:
            tx_seq[tx_id] = s

    with open(args.output, "wb") as f:
        pickle.dump(tx_seq, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(tx_seq)} transcripts to {args.output}")
