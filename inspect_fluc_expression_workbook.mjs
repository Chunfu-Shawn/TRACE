import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/chunfu/Desktop/BGM_lab/translation_model/TE_optimization/Fluc (6h-12h-24h-48h).xlsx";
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const result = await workbook.inspect({
  kind: "workbook,sheet,region",
  maxChars: 20000,
  tableMaxRows: 40,
  tableMaxCols: 20,
  tableMaxCellChars: 120,
});
console.log(result.ndjson);
