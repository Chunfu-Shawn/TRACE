from __future__ import annotations

import copy
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W}


OUTLINE_OLD = [
    "TRACE uses full-length RNA sequence as a shared backbone; bidirectional attention, Rotary Position Embedding, Flash Attention and length-aware batching support variable-length transcripts.",
    "Adaptive Layer Normalization conditions the shared sequence backbone on cellular context.",
    "Training combines position-level occupancy, CDS frame-aware TE and batchwise transcript-ranking objectives. CDS and frame labels supervise training, whereas inference requires no annotations.",
    "On chromosome-held-out transcripts, TRACE produced periodic, CDS-enriched occupancy profiles, indicating generalization of supervised translation patterns.",
    "Predicted periodicity differed between housekeeping transcripts and noncoding RNAs (Fig. 2d).",
    "For ENST00000332859, predicted and observed profiles reached Spearman ρ = 0.80. A localized noncoding-RNA prediction overlapped Ribo-seq evidence without establishing stable protein production.",
    "The same profile supports TE estimation and forward design within known CDSs and annotation-free ORF discovery across unannotated transcripts.",
]

OUTLINE_NEW = [
    "TRACE uses full-length RNA sequence as a shared backbone; bidirectional attention, Rotary Position Embedding, Flash Attention and length-aware batching support variable-length transcripts (Fig. 2a).",
    "Because cis-associated differences accounted for the dominant component of TE variation and cellular context contributed a smaller component, Adaptive Layer Normalization conditioned the shared sequence backbone on cellular context (Fig. 2a). Training combined position-level occupancy, CDS frame-aware TE and batchwise transcript-ranking objectives; CDS and frame labels provided supervision.",
    "On chromosome-held-out transcripts, TRACE generated nucleotide-resolution profiles directly from raw, unannotated RNA sequences and cellular context. Predictions showed three-nucleotide periodicity and CDS enrichment (Fig. 2b,c), indicating generalization of supervised translation patterns.",
    "Predicted periodicity closely agreed with observed periodicity (Spearman ρ = 0.80) but differed between housekeeping transcripts and noncoding RNAs in a manner consistent with their biotypes (Fig. 2d).",
    "Among chromosome-held-out transcripts, TRACE predicted abundant occupancy for a housekeeping transcript with high position-wise agreement to observation (Spearman ρ = 0.80), recovered known polycistronic transcripts, and distinguished lncRNAs with little detectable occupancy from those with localized non-canonical translation-associated signals (Fig. 2e).",
]


MANUSCRIPT_OLD = [
    "Motivated by the substantial contribution of cis sequence suggested by the cross-species analysis, we designed the TRACE backbone to process full-length RNA directly. Bidirectional multi-head self-attention with Rotary Position Embedding captures sequence associations across local motifs and long transcript regions. Flash Attention supports memory-efficient processing of variable-length transcripts. Adaptive Layer Normalization modules introduce cellular context conditioning throughout the transformer stack. This architecture uses RNA sequence as a shared representation and cellular context to modulate nucleotide-resolution predictions (Fig. 2a).",
    "TRACE accepts three inputs: full-length RNA sequence, a species identifier and a cellular context vector derived from gene expression. The RNA contains the 5′ UTR, CDS and 3′ UTR and is tokenized as individual nucleotides. Rotary position embeddings encode 5′-to-3′ position without a fixed-length matrix. Flash Attention and length-aware batching reduce memory and padding costs for transcripts ranging from hundreds to tens of thousands of nucleotides. The output has the same length as the RNA and assigns each position an expected normalized ribosome occupancy (Fig. 2a).",
    "Cellular context was incorporated through Adaptive Layer Normalization modules rather than separate models for each cell type. A low-dimensional embedding of gene expression modulates the scale and shift parameters at each transformer block. The sequence-processing parameters are shared across contexts, whereas conditioning adjusts the predicted occupancy profile for the supplied cellular context.",
    "Training used position-level density, a CDS frame-aware TE objective and a transcript-level ranking objective. Annotated CDS boundaries and reading frames were used to construct the training losses, but were not supplied during inference. Codon identity was represented only through the nucleotide sequence, and RNA-binding-protein motif labels were not provided to the sequence-only model. TRACE is therefore annotation-free at inference: it receives RNA sequence, species and cellular context. Annotations were used only downstream when a benchmark required CDS-level aggregation.",
    "We evaluated TRACE on held-out transcripts from chromosomes 20, 21, 22 and Y. These transcripts and their associated context profiles were excluded from training, model selection and early stopping. During inference, neither CDS nor reading-frame annotation was provided. Predicted profiles showed three-nucleotide periodicity and occupancy concentrated in annotated CDS regions (Fig. 2b,c). Because CDS and frame labels contributed to training, these results indicate generalization of supervised translation patterns to held-out transcripts rather than unsupervised discovery of frame structure.",
    "Annotation-free at inference, TRACE assigned different periodicity scores to transcript classes. Housekeeping transcripts had mean scores above 0.5, whereas long noncoding RNAs had scores below 0.5 (Fig. 2d). This separation is consistent with expected biotype differences, although the training objectives included CDS and reading-frame supervision.",
    "For the housekeeping transcript ENST00000332859, the predicted and observed profiles had a Spearman ρ of 0.80 across nucleotide positions. TRACE predicted little occupancy for the noncoding RNA ENST000000654422. For ENST000000789734, it predicted a localized occupancy region that overlapped an experimentally observed Ribo-seq signal. This example is consistent with coding potential in a transcript annotated as noncoding, but it does not alone establish stable protein production.",
    "TRACE uses one nucleotide-resolution ribosome occupancy profile as a shared translation representation. For a known CDS, the profile can be aggregated to estimate translation efficiency and can score sequence variants for forward design. For an unannotated transcript, the same profile can be scanned for periodic occupancy to support reverse discovery of candidate ORFs. This design connects prediction, design and discovery without separate task-specific representations.",
]

MANUSCRIPT_NEW = [
    "TRACE uses full-length RNA sequence as a shared backbone. Bidirectional multi-head self-attention with Rotary Position Embedding captures sequence associations across local motifs and long transcript regions, while Flash Attention supports memory-efficient processing of variable-length transcripts (Fig. 2a).",
    "TRACE accepts three inputs: full-length RNA sequence, a species identifier and a cellular context vector derived from gene expression. The RNA contains the 5′ UTR, CDS and 3′ UTR and is tokenized as individual nucleotides. Rotary position embeddings encode 5′-to-3′ position without a fixed-length matrix. Flash Attention and length-aware batching reduce memory and padding costs for transcripts ranging from hundreds to tens of thousands of nucleotides. The output has the same length as the RNA and assigns each position an expected normalized ribosome occupancy (Fig. 2a).",
    "Because cis-associated differences accounted for the dominant component of TE variation and cellular context contributed a smaller component, we incorporated cellular context through Adaptive Layer Normalization rather than training a separate model for each cell type (Fig. 2a). A low-dimensional embedding of gene expression modulates the scale and shift parameters at each transformer block. The sequence-processing parameters are shared across contexts, whereas conditioning adjusts the predicted occupancy profile for the supplied cellular context.",
    "Training used position-level occupancy, a CDS frame-aware TE objective and a transcript-level ranking objective. Annotated CDS boundaries and reading frames were used to construct the training losses, but were not supplied during inference. Codon identity was represented only through the nucleotide sequence, and RNA-binding-protein motif labels were not provided to the sequence-only model. TRACE is therefore annotation-free at inference: it receives RNA sequence, species and cellular context. Annotations were used only downstream when a benchmark required CDS-level aggregation.",
    "We evaluated TRACE on held-out transcripts from chromosomes 20, 21, 22 and Y. These transcripts and their associated context profiles were excluded from training, model selection and early stopping. TRACE generated nucleotide-resolution profiles directly from raw, unannotated RNA sequences together with species and cellular context; neither CDS nor reading-frame annotation was provided during inference. Predicted profiles showed three-nucleotide periodicity and occupancy concentrated in annotated CDS regions (Fig. 2b,c). Because CDS and frame labels contributed to training, these results indicate generalization of supervised translation patterns rather than unsupervised discovery of frame structure.",
    "Across chromosome-held-out transcripts, predicted periodicity closely agreed with observed periodicity (Spearman ρ = 0.80). Housekeeping transcripts had mean periodicity scores above 0.5, whereas long noncoding RNAs had scores below 0.5 (Fig. 2d), consistent with their biotypes. This comparison should be interpreted in light of the CDS and reading-frame supervision used during training.",
    "For the housekeeping transcript ENST00000332859, the predicted and observed profiles had a position-wise Spearman ρ of 0.80. TRACE also recovered occupancy patterns in known polycistronic transcripts. It predicted little occupancy for the noncoding RNA ENST000000654422, whereas the prediction for ENST000000789734 contained a localized region overlapping an experimentally observed Ribo-seq signal (Fig. 2e). These examples show that TRACE distinguished lncRNAs with little detectable occupancy from those with localized non-canonical translation-associated signals, although such signals do not alone establish stable protein production.",
]


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def set_paragraph_text(paragraph: etree._Element, text: str) -> None:
    runs = paragraph.xpath("./w:r", namespaces=NS)
    if not runs:
        raise ValueError("Target paragraph has no text run")
    template_run = copy.deepcopy(runs[0])
    for child in list(template_run):
        if child.tag != f"{{{W}}}rPr":
            template_run.remove(child)
    text_node = etree.SubElement(template_run, f"{{{W}}}t")
    if text.startswith(" ") or text.endswith(" "):
        text_node.set(f"{{{XML}}}space", "preserve")
    text_node.text = text
    for child in list(paragraph):
        if child.tag != f"{{{W}}}pPr":
            paragraph.remove(child)
    paragraph.append(template_run)


def revise_document_xml(xml_bytes: bytes, old: list[str], new: list[str]) -> bytes:
    root = etree.fromstring(xml_bytes)
    paragraphs = root.xpath("/w:document/w:body/w:p", namespaces=NS)
    matches = []
    for old_text in old:
        found = [p for p in paragraphs if paragraph_text(p) == old_text]
        if len(found) != 1:
            raise ValueError(f"Expected one matching paragraph, found {len(found)}: {old_text[:80]}")
        matches.append(found[0])

    common = min(len(matches), len(new))
    for paragraph, replacement in zip(matches[:common], new[:common]):
        set_paragraph_text(paragraph, replacement)

    if len(new) > len(matches):
        anchor = matches[-1]
        for replacement in new[len(matches):]:
            added = copy.deepcopy(matches[-1])
            set_paragraph_text(added, replacement)
            anchor.addnext(added)
            anchor = added
    elif len(matches) > len(new):
        for paragraph in matches[len(new):]:
            paragraph.getparent().remove(paragraph)

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def rewrite_docx(source: Path, output: Path, old: list[str], new: list[str]) -> None:
    with zipfile.ZipFile(source, "r") as zin:
        revised_xml = revise_document_xml(zin.read("word/document.xml"), old, new)
        with zipfile.ZipFile(output, "w") as zout:
            for item in zin.infolist():
                payload = revised_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, payload)


def main() -> None:
    source_root = Path("/Users/chunfu/Desktop/BGM_lab/translation_model/manuscript")
    output_root = Path("/Users/chunfu/Desktop/BGM_lab/translation_model/TRACE/.codex_work/result2_revision")
    output_root.mkdir(parents=True, exist_ok=True)
    rewrite_docx(source_root / "TRACE_outline.docx", output_root / "TRACE_outline.docx", OUTLINE_OLD, OUTLINE_NEW)
    rewrite_docx(source_root / "TRACE_manuscript.docx", output_root / "TRACE_manuscript.docx", MANUSCRIPT_OLD, MANUSCRIPT_NEW)


if __name__ == "__main__":
    main()
