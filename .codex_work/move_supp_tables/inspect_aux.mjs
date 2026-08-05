import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("supplementary_tables.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);

for (const [name, range] of [
  ["Comparison", "A1:J15"],
  ["Evidence_notes", "A1:G15"],
  ["Definitions", "A1:B14"],
]) {
  const report = await workbook.inspect({
    kind: "region,computedStyle",
    sheetId: name,
    range,
    maxChars: 10000,
    tableMaxRows: 20,
    tableMaxCols: 12,
  });
  console.log(`--- ${name} ---`);
  console.log(report.ndjson);
  const preview = await workbook.render({ sheetName: name, range, scale: 1.5, format: "png" });
  await fs.writeFile(`${name}.png`, new Uint8Array(await preview.arrayBuffer()));
}
