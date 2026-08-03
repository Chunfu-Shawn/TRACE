from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


SOURCE = Path(".codex_outline_restart/TRACE_outline.source.docx")
OUTPUT = Path(".codex_outline_restart/TRACE_outline.revised.docx")


def main():
    document = Document(SOURCE)
    numbering = document.part.numbering_part.element

    used_num_ids = []
    previous_was_heading = False
    for paragraph in document.paragraphs:
        if paragraph.style.name.startswith("Heading"):
            previous_was_heading = True
            continue
        p_pr = paragraph._p.pPr
        num_pr = p_pr.numPr if p_pr is not None else None
        if previous_was_heading and num_pr is not None:
            num_id = int(num_pr.numId.val)
            if num_id not in used_num_ids:
                used_num_ids.append(num_id)
        if paragraph.text.strip():
            previous_was_heading = False

    for num_id in used_num_ids:
        num = next(
            element
            for element in numbering.findall(qn("w:num"))
            if int(element.get(qn("w:numId"))) == num_id
        )
        for existing in list(num.findall(qn("w:lvlOverride"))):
            if existing.get(qn("w:ilvl")) == "0":
                num.remove(existing)
        override = OxmlElement("w:lvlOverride")
        override.set(qn("w:ilvl"), "0")
        start_override = OxmlElement("w:startOverride")
        start_override.set(qn("w:val"), "1")
        override.append(start_override)
        num.append(override)

    document.save(OUTPUT)


if __name__ == "__main__":
    main()
