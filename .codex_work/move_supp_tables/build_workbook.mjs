import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("supplementary_tables.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);

const sourceEvidence = workbook.worksheets
  .getItem("Evidence_notes")
  .getRange("A4:G15").values;
const sourceDefinitions = workbook.worksheets
  .getItem("Definitions")
  .getRange("A4:B14").values;

const colors = {
  text: "#000000",
  muted: "#595959",
  rule: "#808080",
  header: "#E7E6E6",
  subheader: "#F2F2F2",
  green: "#C6E0B4",
  greenText: "#1B4D1B",
  noFill: "#F2F2F2",
};

function setColumns(sheet, widths) {
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, 1, 1).format.columnWidth = width;
  });
}

function styleTitle(range) {
  range.format = {
    font: { name: "Arial", size: 12, bold: true, color: colors.text },
    verticalAlignment: "center",
    horizontalAlignment: "left",
    wrapText: true,
  };
}

function styleHeader(range, fontSize = 9) {
  range.format = {
    fill: colors.header,
    font: { name: "Arial", size: fontSize, bold: true, color: colors.text },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: {
      top: { style: "medium", color: colors.text },
      bottom: { style: "medium", color: colors.text },
    },
  };
}

function styleBody(range, fontSize = 9) {
  range.format = {
    font: { name: "Arial", size: fontSize, color: colors.text },
    verticalAlignment: "center",
    wrapText: true,
    borders: {
      insideHorizontal: { style: "thin", color: colors.rule },
      bottom: { style: "thin", color: colors.rule },
    },
  };
}

function styleNote(range) {
  range.format = {
    font: { name: "Arial", size: 9, italic: true, color: colors.muted },
    horizontalAlignment: "left",
    verticalAlignment: "top",
    wrapText: true,
  };
}

// Supplementary Table 3: ablation performance.
const table3 = workbook.worksheets.getItem("Table 3");
table3.getRange("A1:Z100").clear({ applyTo: "all" });
table3.showGridLines = false;
table3.mergeCells("A1:G1");
table3.getRange("A1").values = [[
  "Supplementary Table 3 | Ablation performance during validation and zero-shot evaluation",
]];
styleTitle(table3.getRange("A1:G1"));
table3.getRange("A1:G1").format.rowHeight = 25;

table3.getRange("A3:G8").values = [
  [
    "Model",
    "Best loss",
    "RNA-profile Spearman ρ",
    "CDS-mean Spearman ρ",
    "RNA-profile Spearman ρ",
    "CDS-profile Spearman ρ",
    "CDS-mean Spearman ρ",
  ],
  [
    "",
    "Validation for 5 cell-contexts (epoch)",
    "Validation for 5 cell-contexts (epoch)",
    "Validation for 5 cell-contexts (epoch)",
    "Test for 14 unseen cell-contexts",
    "Test for 14 unseen cell-contexts",
    "Test for 14 unseen cell-contexts",
  ],
  ["TRACE-Real", "0.1067 (29)", "0.4459 (47)", "0.6487 (41)", 0.4381, 0.2908, 0.3432],
  ["TRACE", "0.1077 (41)", "0.4441 (40)", "0.6345 (42)", 0.4430, 0.3095, 0.3590],
  ["LN", "0.1109 (42)", "0.4270 (10)", "0.6001 (44)", 0.4386, 0.2922, 0.3431],
  ["Conv", "0.1190 (5)", "0.3987 (5)", "0.5867 (40)", 0.3604, 0.3061, 0.2911],
];
styleHeader(table3.getRange("A3:G3"));
table3.getRange("A4:G4").format = {
  fill: colors.subheader,
  font: { name: "Arial", size: 8, bold: true, color: colors.muted },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { bottom: { style: "medium", color: colors.text } },
};
styleBody(table3.getRange("A5:G8"));
table3.getRange("A5:A8").format.font = { name: "Arial", size: 9, bold: true };
table3.getRange("B5:G8").format.horizontalAlignment = "center";
table3.getRange("E5:G8").format.numberFormat = "0.0000";
table3.getRange("B5:D5").format.fill = colors.green;
table3.getRange("E6:G6").format.fill = colors.green;
table3.getRange("A3:G3").format.rowHeight = 32;
table3.getRange("A4:G4").format.rowHeight = 34;
table3.getRange("A5:G8").format.rowHeight = 23;

table3.mergeCells("A10:G10");
table3.getRange("A10").values = [[
  "Table note. Validation values are the best values recorded independently for each metric across 50 training epochs on the human_5c_6k_depth0.1_cov0.1_rpm1 validation dataset; the corresponding epoch is shown in parentheses. Test values are equal-context means across the 14 eligible cellular contexts among 26 unseen human cell lines in the chromosome-held-out test dataset, using the checkpoint selected by the lowest validation loss. Eligibility required at least 800 retained transcripts with RPF depth of at least 0.1 per context. Higher Spearman ρ and lower validation loss indicate better performance. The best value in each column is shaded green. TRACE-Real achieved the best validation values, whereas TRACE achieved the best zero-shot test values, consistent with expression masking and interpolation improving generalization to unseen cellular contexts. TRACE corresponds to TRACE-Augment in the source result files.",
]];
styleNote(table3.getRange("A10:G10"));
table3.getRange("A10:G10").format.rowHeight = 88;
setColumns(table3, [17, 21, 25, 24, 24, 24, 23]);
table3.freezePanes.freezeRows(4);

// Supplementary Table 4: capability comparison and supporting notes.
const table4 = workbook.worksheets.add("Table 4");
table4.showGridLines = false;
table4.mergeCells("A1:J1");
table4.getRange("A1").values = [[
  "Supplementary Table 4 | Capability comparison of TRACE with prior translation models and ORF-identification tools",
]];
styleTitle(table4.getRange("A1:J1"));
table4.getRange("A1:J1").format.rowHeight = 25;

const comparison = [
  ["Model", "Full-length RNA sequence", "Variable-length input", "CDS-annotation-free inference", "Ribo-seq-free inference", "Single-nucleotide signal output", "Cell-context conditioning", "Unified multi-species modeling", "Direct TE prediction", "Direct ORF identification"],
  ["TRACE", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
  ["Translatomer", "No", "No", "Yes", "Yes", "No", "Yes", "No", "No", "No"],
  ["RiboMIMO", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No"],
  ["Riboformer", "No", "No", "No", "No", "No", "No", "No", "No", "No"],
  ["Optimus 5-Prime", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No"],
  ["RiboDecode", "No", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
  ["RiboNN", "Yes", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
  ["RiboTIE", "Yes", "Yes", "Yes", "No", "Yes", "No", "No", "No", "Yes"],
  ["Ribo-TISH", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "Yes"],
  ["RibORF", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "Yes"],
  ["TranslationAI", "Yes", "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "Yes"],
];
table4.getRange("A3:J14").values = comparison;
styleHeader(table4.getRange("A3:J3"), 8);
styleBody(table4.getRange("A4:J14"));
table4.getRange("A4:A14").format.font = { name: "Arial", size: 9, bold: true };
table4.getRange("B4:J14").format.horizontalAlignment = "center";
table4.getRange("A3:J3").format.rowHeight = 48;
table4.getRange("A4:J14").format.rowHeight = 22;

for (let row = 4; row <= 14; row += 1) {
  for (let col = 2; col <= 10; col += 1) {
    const cell = table4.getRangeByIndexes(row - 1, col - 1, 1, 1);
    if (cell.values[0][0] === "Yes") {
      cell.format = {
        fill: colors.green,
        font: { name: "Arial", size: 9, bold: true, color: colors.greenText },
        horizontalAlignment: "center",
        verticalAlignment: "center",
      };
    } else {
      cell.format.fill = colors.noFill;
      cell.format.font = { name: "Arial", size: 9, color: colors.muted };
    }
  }
}

table4.mergeCells("A16:J16");
table4.getRange("A16").values = [[
  "Table note. Yes denotes direct support under the operational definitions below; No denotes that the capability is absent or is available only indirectly, through a fixed local window, a separate species-specific model, task-specific retraining, post hoc analysis or a required Ribo-seq input. All columns are phrased as positive capabilities, so Yes under Ribo-seq-free inference means that raw or processed Ribo-seq data are not required at inference. A No designation therefore indicates a difference in scope rather than lower overall model quality.",
]];
styleNote(table4.getRange("A16:J16"));
table4.getRange("A16:J16").format.rowHeight = 58;

table4.mergeCells("A18:J18");
table4.getRange("A18").values = [["Evidence notes"]];
table4.getRange("A18:J18").format = {
  font: { name: "Arial", size: 10, bold: true, color: colors.text },
  borders: { bottom: { style: "medium", color: colors.text } },
};
table4.getRange("A19:G30").values = sourceEvidence;
styleHeader(table4.getRange("A19:G19"), 8);
styleBody(table4.getRange("A20:G30"), 8);
table4.getRange("A19:G19").format.rowHeight = 34;
table4.getRange("A20:G30").format.rowHeight = 72;

table4.mergeCells("A32:J32");
table4.getRange("A32").values = [["Definitions and classification rules"]];
table4.getRange("A32:J32").format = {
  font: { name: "Arial", size: 10, bold: true, color: colors.text },
  borders: { bottom: { style: "medium", color: colors.text } },
};
table4.getRange("A33").values = [["Term"]];
table4.mergeCells("B33:J33");
table4.getRange("B33").values = [["Operational definition"]];
styleHeader(table4.getRange("A33:J33"), 8);

const definitions = sourceDefinitions.slice(1);
for (let index = 0; index < definitions.length; index += 1) {
  const row = 34 + index;
  const [term, originalDefinition] = definitions[index];
  let definition = originalDefinition;
  if (term === "Scope") {
    definition = "The binary comparison follows Supplementary Table 4 and the operational definitions listed below.";
  }
  if (term === "Single-nt ribosome-density output") {
    definition = "Requires a predicted ribosome-occupancy value at each nucleotide; TIS/TTS probabilities and codon-level outputs do not qualify.";
  }
  if (term === "Explicit cell-context conditioning") {
    definition = "Requires a cell-environment representation supplied at inference; assignments follow Supplementary Table 4.";
  }
  const displayedTerm = term === "Single-nt ribosome-density output"
    ? "Single-nucleotide signal output"
    : term;
  table4.getRange(`A${row}`).values = [[displayedTerm]];
  table4.mergeCells(`B${row}:J${row}`);
  table4.getRange(`B${row}`).values = [[definition]];
  styleBody(table4.getRange(`A${row}:J${row}`), 8);
  table4.getRange(`A${row}`).format.font = { name: "Arial", size: 8, bold: true };
  table4.getRange(`A${row}:J${row}`).format.rowHeight = 34;
}

table4.mergeCells("A45:J45");
table4.getRange("A45").values = [[
  "Evidence note. Feature assignments were checked against the primary publications and official implementations summarized above.",
]];
styleNote(table4.getRange("A45:J45"));
table4.getRange("A45:J45").format.rowHeight = 26;

setColumns(table4, [18, 24, 38, 38, 28, 30, 22, 24, 20, 20]);
table4.freezePanes.freezeRows(3);

const table3Preview = await workbook.render({
  sheetName: "Table 3",
  range: "A1:G10",
  scale: 1.4,
  format: "png",
});
await fs.writeFile("Table_3_final.png", new Uint8Array(await table3Preview.arrayBuffer()));

const table4MainPreview = await workbook.render({
  sheetName: "Table 4",
  range: "A1:J16",
  scale: 1.2,
  format: "png",
});
await fs.writeFile("Table_4_main_final.png", new Uint8Array(await table4MainPreview.arrayBuffer()));

const table4NotesPreview = await workbook.render({
  sheetName: "Table 4",
  range: "A18:J45",
  scale: 1.0,
  format: "png",
});
await fs.writeFile("Table_4_notes_final.png", new Uint8Array(await table4NotesPreview.arrayBuffer()));

try {
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save("supplementary_tables.updated.xlsx");
} catch (error) {
  console.error("EXPORT_ERROR", error?.name, error?.message);
  process.exitCode = 1;
}
