NEAR_COGNATE_START_CODONS = {'CTG', 'GTG', 'TTG', 'ACG'}


def normalize_initiator_codon(sequence: str) -> str:
    """Replace only a near-cognate ORF initiator with ATG before translation."""
    sequence = str(sequence).upper().replace('U', 'T')
    if len(sequence) >= 3 and sequence[:3] in NEAR_COGNATE_START_CODONS:
        return 'ATG' + sequence[3:]
    return sequence
