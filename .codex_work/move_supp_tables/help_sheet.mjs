import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("supplementary_tables.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);
console.log(
  workbook.help("worksheet operations", {
    search: "delete|remove|rename|name",
    include: "index,examples,notes",
    maxChars: 5000,
  }).ndjson,
);
