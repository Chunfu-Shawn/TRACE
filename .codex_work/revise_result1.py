from __future__ import annotations

import copy
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W}


OUTLINE_OLD = [
    "Raw Ribo-seq signal is sparse, protocol dependent and confounded by transcript abundance.",
    "We curated 1,260 Ribo-seq runs and 710 matched RNA-seq runs from 73 cellular contexts across human, rhesus macaque and mouse.",
    "A unified workflow performed quality control, within-context merging, representative-transcript selection, nuclease-aware P-site inference, and RNA-abundance normalization.",
    "Within-context merging substantially increased per-transcript coverage, with CDS coverage of housekeeping genes approaching 1.0.",
    "Compositional regression reduced RNA-abundance confounding to yield relative ribosome occupancy profiles.",
    "Cross-species and cross-context patterns are consistent with a substantial cis-sequence contribution without establishing causal dominance. This supports an RNA-sequence backbone with cellular context conditioning.",
    "Processed profiles were enriched across annotated CDSs and showed strong three-nucleotide periodicity.",
    "The final resource contains 1.8 million transcript–context profiles, with one profile for each observed transcript and cellular context",
    "The resource includes protein-coding transcripts and noncoding RNAs with measurable occupancy but does not represent every translatome state.",
]

OUTLINE_NEW = [
    "To enable simultaneous modeling of multiple translation tasks, we constructed a cross-context translation dataset in which transcript-level translation efficiency and nucleotide-resolution translation dynamics are represented within the same harmonized profile.",
    "We curated 1,260 Ribo-seq runs and 710 matched RNA-seq runs from 73 cellular contexts across human, rhesus macaque and mouse.",
    "A unified workflow performed quality control, within-context merging, representative-transcript selection, nuclease-aware P-site inference, and RNA-abundance normalization (Fig. 1a).",
    "Within-context merging substantially increased per-transcript coverage, with CDS coverage of housekeeping genes approaching 1.0 (Fig. 1b).",
    "To capture non-canonical translation, transcript retention was independent of CDS annotation. We retained transcripts with sufficient ribosome occupancy signal; 8.3% of annotated lncRNAs genome-wide showed detectable occupancy in at least one cellular context (Fig. 1c), expanding the diversity of translation-associated patterns available for model training.",
    "Because experimental conditions varied across Ribo-seq datasets, a unified processing workflow aligned P-site assignments across protocols. The resulting profiles showed a consistent initiation peak at annotated CDS starts and phase-aligned periodic signals, with three-nucleotide periodicity scores exceeding 0.5 (Fig. 1d), providing coherent nucleotide-resolution supervision.",
    "Ribo-seq abundance is strongly influenced by transcript abundance and varies across cellular contexts. Compositional regression reduced RNA-abundance confounding to yield relative ribosome occupancy profiles. The resulting TE distributions varied less across cellular contexts (Fig. 1e), whereas substantial differences remained among the three species (Fig. 1f).",
    "The final resource contains 1.8 million transcript–context profiles, with one profile for each observed transcript and cellular context (Extended Data Fig. 1a). Each harmonized profile jointly represents transcript-level TE and nucleotide-resolution translation dynamics.",
]


MANUSCRIPT_OLD = [
    "The construction of a reliable training dataset proved to be a non-trivial component of this work. Raw Ribo-seq data present several challenges for quantitative modeling. First, read coverage per transcript is typically sparse, with many nucleotide positions receiving zero counts, particularly in transcripts with low ribosome occupancy. Second, experimental protocols differ across laboratories in nuclease choice, size selection, and library preparation, introducing systematic biases in the distribution of ribosome-protected fragment lengths. Third, Ribo-seq signal conflates translational activity with transcript abundance: a highly expressed transcript with modest translation efficiency can produce more Ribo-seq reads than a lowly expressed transcript that is efficiently translated. Disentangling these factors required careful preprocessing.",
    "We aggregated public Ribo-seq and matched RNA-seq datasets covering human, rhesus macaque, and mouse. After curation, the collection comprised 1,355 Ribo-seq runs and 475 RNA-seq runs from 73 cellular contexts, together with structured metadata recording species, tissue of origin, and experimental conditions.",
    "All data passed through a common workflow. After run-level quality control, BAM files from the same cellular context were merged. Representative transcripts were selected from alignment support; exon bins with at least two reads were considered informative, MANE Select transcripts were retained, and non-MANE isoforms adding no informative exon or splice-junction evidence were removed. A nuclease-aware LightGBM classifier then assigned read-specific P-sites across library conditions. Profiles were subsequently filtered by RNA abundance, P-site depth, coverage and, for annotated CDSs, frame-0 enrichment. Dataset-level periodicity quality control retained libraries with clear three-nucleotide signal. Detailed procedures are provided in Supplementary Methods.",
    "To isolate translational signal from RNA abundance, we applied compositional regression to estimate a translation efficiency scaling factor for each transcript. Multiplying normalized P-site profiles by this factor yielded nucleotide-resolution ribosome occupancy profiles after accounting for expression level. The dataset comprises 1.8 million transcript–context profiles. A transcript contributes one profile for each cellular context with an available ribosome-density profile. Canonical protein-coding transcripts showed dense, periodic occupancy within annotated CDS regions.",
    "Following transcriptional normalization, variation in translation efficiency across transcripts exceeded variation among cellular contexts. Between-species differences also exceeded within-species differences among cellular contexts. These patterns are consistent with a substantial contribution of cis sequence to translational output. They do not establish causal dominance because species differ in many sequence-independent factors. Related cellular contexts nevertheless clustered by translation efficiency across species, supporting an additional contribution from trans-acting cellular programs. We therefore used RNA sequence as the model backbone and cellular context as a conditioning signal for context-specific modulation (Fig. 1).",
]

MANUSCRIPT_NEW = [
    "To enable joint modeling of multiple translation tasks, we constructed a cross-context translation dataset in which transcript-level translation efficiency (TE) and nucleotide-resolution translation dynamics are represented within the same harmonized profile. This design provides a common target for TE estimation, local occupancy prediction and ORF identification.",
    "We aggregated public Ribo-seq and matched RNA-seq datasets covering human, rhesus macaque and mouse. After curation, the collection comprised 1,260 Ribo-seq runs and 710 matched RNA-seq runs from 73 cellular contexts, together with structured metadata recording species, tissue of origin and experimental conditions.",
    "All data passed through a unified workflow encompassing run-level quality control, within-context merging, representative-transcript selection, nuclease-aware P-site inference, profile filtering and RNA-abundance normalization (Fig. 1a). BAM files from the same cellular context were merged, substantially increasing per-transcript coverage; CDS coverage of housekeeping genes approached 1.0 (Fig. 1b). Detailed procedures are provided in Supplementary Methods.",
    "To retain signals from non-canonical translation, transcript inclusion did not depend on CDS annotation. Representative transcripts were selected from Ribo-seq alignment support, and transcripts with sufficient P-site depth and coverage were retained. Across the genome, 8.3% of annotated lncRNAs showed detectable ribosome occupancy in at least one cellular context (Fig. 1c). Including these transcripts expanded the diversity of translation-associated patterns available to the model.",
    "Ribo-seq studies used different nucleases and library conditions, producing protocol-specific distributions of ribosome-protected fragment lengths. We therefore applied a unified, nuclease-aware procedure to align read-specific P-site assignments across datasets. After processing, the profiles displayed a consistent occupancy peak at annotated CDS starts and phase-aligned periodic signals, with three-nucleotide periodicity scores exceeding 0.5 (Fig. 1d). These profiles provided coherent nucleotide-resolution supervision across heterogeneous experiments.",
    "Raw Ribo-seq abundance is confounded by transcript abundance and varies among cellular contexts. Compositional regression reduced RNA-abundance confounding to yield relative ribosome occupancy profiles. After normalization, TE varied less among cellular contexts (Fig. 1e), whereas marked differences persisted among human, rhesus macaque and mouse (Fig. 1f).",
    "The final dataset comprises 1.8 million transcript–context profiles, with one profile for each observed transcript and cellular context (Extended Data Fig. 1a). Each harmonized profile jointly represents transcript-level TE and nucleotide-resolution translation dynamics.",
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
    output_root = Path("/Users/chunfu/Desktop/BGM_lab/translation_model/TRACE/.codex_work/result1_revision")
    output_root.mkdir(parents=True, exist_ok=True)
    rewrite_docx(source_root / "TRACE_outline.docx", output_root / "TRACE_outline.docx", OUTLINE_OLD, OUTLINE_NEW)
    rewrite_docx(source_root / "TRACE_manuscript.docx", output_root / "TRACE_manuscript.docx", MANUSCRIPT_OLD, MANUSCRIPT_NEW)


if __name__ == "__main__":
    main()
