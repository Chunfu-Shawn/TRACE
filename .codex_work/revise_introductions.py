from __future__ import annotations

import copy
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W}


OUTLINE_OLD = [
    "Translation varies across transcripts and cellular contexts, so transcript abundance alone cannot predict protein output. Nucleotide-resolution occupancy models could support both sequence design and translated-ORF discovery.",
    "Ribo-seq provides nucleotide-resolution occupancy but is difficult to deploy broadly. Existing models usually separate TE estimation, local dynamics and ORF identification, and often require CDS annotations, fixed windows or Ribo-seq input.",
    "A unified model must capture multiscale sequence features and cellular context while learning from heterogeneous Ribo-seq data. We address this with harmonized targets, variable-length modeling, context conditioning and chromosome-held-out evaluation.",
    "TRACE predicts full-length occupancy profiles and is annotation-free at inference. Trained on 1.8 million transcript–context profiles from 73 cellular contexts across three species, it unifies translation dynamics, TE, ORF identification, cultured-cell design and neoantigen prioritization (Supplementary Table 3).",
]

OUTLINE_NEW = [
    "Gene expression is a multistep process, and protein abundance correlates only moderately with transcript abundance (typically r = 0.4–0.6), with substantial variation among cellular contexts. Translational regulation is a major contributor, shaping protein-specific translation efficiency (TE) and the production of non-canonical small peptides.",
    "Ribosome profiling (Ribo-seq) provides precise evidence and quantitative measurements of translation, but demanding sample requirements, complex workflows, sparse read coverage and high cost limit its large-scale application to rare and diseased tissues.",
    "Computational models have therefore been developed to predict translation efficiency and dynamics from RNA. Many rely primarily on convolutional architectures and lack nucleotide-resolution outputs, limiting their ability to capture long-range dependencies across full-length RNAs and fine-scale translation patterns.",
    "Prior translation models are also largely task specific, addressing ORF identification, translation-efficiency estimation or translation-dynamics prediction separately, although these processes are biologically coupled. A unified model is hindered by the lack of a harmonized dataset spanning these outputs. TE data are increasingly available through resources such as RiboBase, but harmonized nucleotide-level translation signals remain scarce.",
    "We developed TRACE, a transformer with Adaptive Layer Normalization–Zero (AdaLN-Zero) trained on a newly constructed harmonized dataset. TRACE predicts nucleotide-resolution ribosome occupancy profiles for full-length RNAs across cellular contexts and is annotation-free at inference. Its shared output supports TE estimation and translation-dynamics prediction for known CDSs, coding-ORF identification in unannotated RNAs, and applications in mRNA therapeutics, vaccines and tumor neoantigen discovery.",
]

MANUSCRIPT_OLD = [
    "Protein output cannot be inferred reliably from transcript abundance alone because translation varies across transcripts and cellular contexts. Ribosome profiling has revealed translated regions in canonical coding transcripts and in many RNAs previously classified as noncoding. These observations expand the biological scope of translation and motivate predictive models that resolve where ribosomes occupy each transcript. Such models could support both sequence design and the discovery of translated open reading frames (ORFs).",
    "Ribosome profiling measures ribosome occupancy at nucleotide resolution, but its experimental requirements limit coverage across tissues, cell states and clinical samples. Existing computational methods address parts of this problem. Some estimate transcript-level translation efficiency or local CDS density, whereas others identify ORFs from Ribo-seq or predict translation boundaries from sequence. However, these capabilities are usually separated across models, and several approaches require CDS annotations, fixed sequence windows or experimental Ribo-seq inputs. A unified model must predict a full-length occupancy profile while remaining annotation-free at inference.",
    "This task is challenging for three reasons. First, translation depends on sequence features ranging from local initiation motifs to full-transcript architecture. Second, the same RNA can be translated differently across cellular contexts. Third, training data from different Ribo-seq studies vary in coverage, protocol and P-site assignment. Addressing these challenges requires a harmonized dataset, a variable-length sequence model and explicit cellular context conditioning. It also requires evaluation on held-out transcripts that do not overlap the training set.",
    "Here we present TRACE (Translation Resolution Across Cell Environments), a transformer that predicts a nucleotide-resolution ribosome occupancy profile from full-length RNA sequence and cellular context. TRACE was trained on 1.8 million transcript–context profiles from 73 cellular contexts across three species. Among the methods compared in Supplementary Table 3, only TRACE combines full-length variable-length input, cellular context conditioning, unified multi-species modeling, nucleotide-resolution occupancy prediction and annotation-free inference. This translation representation supports tasks that are commonly separated. Within a known CDS, occupancy magnitude and position provide an objective for forward sequence design. Across an unannotated transcript, the same profile supports reverse discovery by localizing candidate translated ORFs. We evaluate these capabilities across translation dynamics, translation efficiency, ORF identification, cultured-cell mRNA design and tumor neoantigen candidate prioritization.",
]

MANUSCRIPT_NEW = [
    "Gene expression is a multistep process in which transcript abundance provides only a partial account of protein output. Across published datasets, correlations between mRNA and protein abundance are typically 0.4–0.6 and vary substantially among cellular contexts. Translational regulation is therefore a major layer of gene-expression control. By modulating ribosome engagement and translation efficiency (TE), it helps establish protein-specific output and enables the production of non-canonical small peptides from regions outside conventional CDS annotations.",
    "Ribosome profiling (Ribo-seq) provides nucleotide-resolution evidence of translation and quantitative measurements of ribosome occupancy. However, it requires high-quality input material, involves complex experimental and computational workflows, often yields sparse read coverage with extensive dropout, and remains costly. These constraints hinder its large-scale application to rare cell populations, scarce tissues and disease specimens.",
    "These limitations have motivated computational models that predict translation efficiency and translation dynamics from RNA sequence. Many existing models rely primarily on convolutional modules and do not produce nucleotide-resolution profiles. Consequently, they are not designed to jointly capture long-range dependencies across full-length RNAs and fine-scale local translation patterns.",
    "A second limitation is fragmentation by task. Prior translation models typically address ORF identification, translation-efficiency estimation or translation-dynamics prediction separately, although these quantities arise from a coupled biological process and influence one another. A unified model has also been constrained by the absence of a harmonized dataset spanning these outputs. Transcript-level TE measurements have been aggregated in resources such as RiboBase, but harmonized nucleotide-level translation signals remain scarce.",
    "Here we present TRACE (Translation Resolution Across Cell Environments), a transformer with Adaptive Layer Normalization–Zero (AdaLN-Zero) conditioning trained on a newly constructed harmonized dataset of 1.8 million transcript–context profiles from 73 cellular contexts across three species. From full-length RNA sequence and cellular context, TRACE predicts a nucleotide-resolution ribosome occupancy profile and is annotation-free at inference. The same translation representation supports TE estimation and translation-dynamics prediction for known CDSs and coding-ORF identification in unannotated RNAs. By unifying tasks conventionally treated separately, TRACE provides a computational framework for therapeutic mRNA design, vaccine development and tumor neoantigen discovery.",
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

    for paragraph, replacement in zip(matches, new[: len(old)]):
        set_paragraph_text(paragraph, replacement)

    added = copy.deepcopy(matches[-1])
    set_paragraph_text(added, new[-1])
    matches[-1].addnext(added)

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def rewrite_docx(source: Path, output: Path, old: list[str], new: list[str]) -> None:
    with zipfile.ZipFile(source, "r") as zin:
        document_xml = zin.read("word/document.xml")
        revised_xml = revise_document_xml(document_xml, old, new)
        with zipfile.ZipFile(output, "w") as zout:
            for item in zin.infolist():
                payload = revised_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, payload)


def main() -> None:
    source_root = Path("/Users/chunfu/Desktop/BGM_lab/translation_model/manuscript")
    output_root = Path("/Users/chunfu/Desktop/BGM_lab/translation_model/TRACE/.codex_work/intro_revision")
    output_root.mkdir(parents=True, exist_ok=True)
    rewrite_docx(source_root / "TRACE_outline.docx", output_root / "TRACE_outline.docx", OUTLINE_OLD, OUTLINE_NEW)
    rewrite_docx(source_root / "TRACE_manuscript.docx", output_root / "TRACE_manuscript.docx", MANUSCRIPT_OLD, MANUSCRIPT_NEW)


if __name__ == "__main__":
    main()
