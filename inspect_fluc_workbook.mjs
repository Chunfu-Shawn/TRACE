import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/chunfu/Desktop/BGM_lab/translation_model/TE_optimization/Fluc_order_list_final.xlsx";
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const result = await workbook.inspect({
  kind: "workbook,sheet,region",
  maxChars: 12000,
  tableMaxRows: 12,
  tableMaxCols: 12,
  tableMaxCellChars: 180,
});
console.log(result.ndjson);
