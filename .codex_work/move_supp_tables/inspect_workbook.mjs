import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("supplementary_tables.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);
const sheets = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
console.log(sheets.ndjson);

for (const name of ["Table 1", "Table 2"]) {
  const preview = await workbook.render({
    sheetName: name,
    range: "A1:K20",
    scale: 1.5,
    format: "png",
  });
  await fs.writeFile(
    `${name.replaceAll(" ", "_")}.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}
