import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/chunfu/Desktop/BGM_lab/translation_model/TE_optimization/Fluc_order_list_final.xlsx";
const outputPath = "/Users/chunfu/Desktop/BGM_lab/translation_model/TRACE/Fluc_order_list_final_cDNA.fasta";

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItem("Fluc_order_list_V1");
const values = sheet.getRange("A1:D15").values;

const expectedHeaders = ["ID", "5'UTR", "CDS", "3'UTR"];
if (JSON.stringify(values[0]) !== JSON.stringify(expectedHeaders)) {
  throw new Error(`Unexpected headers: ${JSON.stringify(values[0])}`);
}

const records = [];
const seenIds = new Set();
for (const row of values.slice(1)) {
  const id = String(row[0] ?? "").trim();
  if (!id) continue;
  if (seenIds.has(id)) throw new Error(`Duplicate ID: ${id}`);
  seenIds.add(id);

  const segments = row.slice(1, 4).map((value) =>
    String(value ?? "").toUpperCase().replace(/U/g, "T").replace(/\s+/g, ""),
  );
  const sequence = segments.join("");
  if (!sequence) throw new Error(`Empty sequence: ${id}`);
  const invalid = [...new Set(sequence.replace(/[ACGTRYSWKMBDHVN]/g, ""))];
  if (invalid.length) {
    throw new Error(`Invalid sequence characters for ${id}: ${invalid.join("")}`);
  }
  records.push({ id, sequence, segmentLengths: segments.map((segment) => segment.length) });
}

const wrap = (sequence, width = 80) =>
  sequence.match(new RegExp(`.{1,${width}}`, "g")).join("\n");
const fasta = `${records.map(({ id, sequence }) => `>${id}\n${wrap(sequence)}`).join("\n")}\n`;
await fs.writeFile(outputPath, fasta, "utf8");

console.log(JSON.stringify({
  outputPath,
  recordCount: records.length,
  records: records.map(({ id, sequence, segmentLengths }) => ({
    id,
    length: sequence.length,
    utr5Length: segmentLengths[0],
    cdsLength: segmentLengths[1],
    utr3Length: segmentLengths[2],
  })),
}, null, 2));
