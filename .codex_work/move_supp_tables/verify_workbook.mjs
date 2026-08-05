import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("supplementary_tables.updated.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);

for (const [sheetId, range] of [
  ["Table 3", "A1:G10"],
  ["Table 4", "A1:J45"],
]) {
  const report = await workbook.inspect({
    kind: "region",
    sheetId,
    range,
    maxChars: 8000,
    tableMaxRows: 50,
    tableMaxCols: 12,
  });
  console.log(report.ndjson);
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);
