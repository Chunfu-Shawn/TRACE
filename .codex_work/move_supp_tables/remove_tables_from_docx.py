from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph


SOURCE = Path(__file__).with_name("supplementary_information.docx")
OUTPUT = SOURCE.with_name("supplementary_information.updated.docx")

document = Document(SOURCE)
body = document._element.body

section_break_paragraph = next(
    paragraph
    for paragraph in document.paragraphs
    if paragraph.text.startswith("Validation-selected checkpoints were evaluated")
)
paragraph_properties = section_break_paragraph._p.get_or_add_pPr()
portrait_section_properties = paragraph_properties.find(qn("w:sectPr"))
if portrait_section_properties is None:
    raise RuntimeError("Expected the portrait section break after Supplementary Methods")

final_section_properties = body.find(qn("w:sectPr"))
if final_section_properties is None:
    raise RuntimeError("Expected final section properties")
body.replace(final_section_properties, deepcopy(portrait_section_properties))
paragraph_properties.remove(portrait_section_properties)

remove_mode = False
removed_tables = 0
removed_paragraphs = 0
for element in list(body.iterchildren()):
    if element.tag == qn("w:sectPr"):
        continue
    if element.tag == qn("w:p"):
        paragraph = Paragraph(element, document)
        if paragraph.text.startswith("Supplementary Table 2 |"):
            remove_mode = True
    if remove_mode:
        if element.tag == qn("w:tbl"):
            removed_tables += 1
        elif element.tag == qn("w:p"):
            removed_paragraphs += 1
        body.remove(element)

if removed_tables != 2:
    raise RuntimeError(f"Expected to remove two tables, removed {removed_tables}")
if removed_paragraphs != 6:
    raise RuntimeError(f"Expected to remove six table-related paragraphs, removed {removed_paragraphs}")

document.save(OUTPUT)
print(OUTPUT)
