from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


SOURCE = Path(__file__).with_name("TRACE_outline.docx")
OUTPUT = SOURCE.with_name("TRACE_outline.updated.docx")

document = Document(SOURCE)
numbering = document.part.numbering_part.element


def get_num_id(paragraph):
    properties = paragraph._p.pPr
    if properties is None:
        return None
    numbering_properties = properties.find(qn("w:numPr"))
    if numbering_properties is None:
        return None
    num_id = numbering_properties.find(qn("w:numId"))
    if num_id is None:
        return None
    return int(num_id.get(qn("w:val")))


def allocate_num_id(source_num_id):
    existing_ids = [
        int(element.get(qn("w:numId")))
        for element in numbering.findall(qn("w:num"))
    ]
    new_num_id = max(existing_ids, default=0) + 1
    source_num = next(
        element
        for element in numbering.findall(qn("w:num"))
        if int(element.get(qn("w:numId"))) == source_num_id
    )
    abstract_num_id = source_num.find(qn("w:abstractNumId"))
    if abstract_num_id is None:
        raise RuntimeError(f"Numbering definition {source_num_id} has no abstractNumId")

    new_num = OxmlElement("w:num")
    new_num.set(qn("w:numId"), str(new_num_id))
    new_num.append(deepcopy(abstract_num_id))
    level_override = OxmlElement("w:lvlOverride")
    level_override.set(qn("w:ilvl"), "0")
    start_override = OxmlElement("w:startOverride")
    start_override.set(qn("w:val"), "1")
    level_override.append(start_override)
    new_num.append(level_override)
    numbering.append(new_num)
    return new_num_id


section_headings = {
    "INTRODUCTION",
    "RESULT 1: A nucleotide-resolution dataset of ribosome occupancy across cellular contexts",
    "RESULT 2: TRACE predicts nucleotide-resolution ribosome occupancy profiles",
    "RESULT 3: A unified translation representation supports three benchmark tasks",
    "RESULT 4: TRACE predictions show translation-associated patterns across sequence scales",
    "RESULT 5: Forward design of CDS translation in specific cellular contexts",
    "RESULT 6: Reverse discovery of tumor neoantigen candidates from transcriptomes",
    "DISCUSSION",
}

current_num_id = None
source_num_id = None
sections_updated = 0
numbered_paragraphs = 0

for paragraph in document.paragraphs:
    if paragraph.text in section_headings:
        current_num_id = None
        sections_updated += 1
        continue

    paragraph_num_id = get_num_id(paragraph)
    if paragraph_num_id is None:
        continue
    if source_num_id is None:
        source_num_id = paragraph_num_id
    if current_num_id is None:
        current_num_id = (
            paragraph_num_id
            if sections_updated == 1
            else allocate_num_id(source_num_id)
        )

    numbering_properties = paragraph._p.pPr.find(qn("w:numPr"))
    num_id_element = numbering_properties.find(qn("w:numId"))
    num_id_element.set(qn("w:val"), str(current_num_id))
    numbered_paragraphs += 1

if sections_updated != 8:
    raise RuntimeError(f"Expected eight numbered sections, found {sections_updated}")
if numbered_paragraphs != 44:
    raise RuntimeError(
        f"Expected 44 numbered key-point paragraphs, found {numbered_paragraphs}"
    )

document.save(OUTPUT)
print(OUTPUT)
