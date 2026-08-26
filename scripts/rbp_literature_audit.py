#!/usr/bin/env python3
"""Build a reproducible literature audit for RBP motif-effect results.

The script extracts the RBP labels shown in a Matplotlib PDF, joins their
region-specific model effects, queries Europe PMC or OpenAlex, and writes
paper-level and RBP-level screening tables.
It intentionally treats effect direction inferred from abstracts as a
screening annotation rather than definitive mechanistic evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader


TRANSLATION_TERMS = (
    "translation", "translational", "ribosome", "ribosomal", "polysome",
    "protein synthesis", "translation initiation", "ires",
)
SPLICING_TERMS = (
    "splicing", "spliceosome", "spliceosomal", "pre-mrna", "pre‐mrna",
    "intron retention", "retained intron", "exon inclusion", "exon skipping",
)
NUCLEAR_TERMS = (
    "nuclear retention", "nuclear export", "nucleocytoplasmic",
    "nucleo-cytoplasmic", "rna export", "mrna export",
)
STABILITY_TERMS = (
    "rna stability", "mrna stability", "rna decay", "mrna decay",
    "deadenylation", "degradation", "destabiliz", "stabiliz",
)
LNCRNA_TERMS = ("lncrna", "long noncoding rna", "long non-coding rna")
POSITIVE_TERMS = (
    "promotes", "enhances", "stimulates", "activates", "increases",
    "upregulates", "up-regulates", "stabilizes", "facilitates",
)
NEGATIVE_TERMS = (
    "represses", "inhibits", "suppresses", "decreases", "downregulates",
    "down-regulates", "destabilizes", "degrades", "attenuates",
)
REGION_PATTERNS = {
    "5UTR": ("5' utr", "5′ utr", "5utr", "5' untranslated", "5′ untranslated"),
    "CDS/exon": ("coding sequence", "cds", "open reading frame", "exonic", "exon"),
    "3UTR": ("3' utr", "3′ utr", "3utr", "3' untranslated", "3′ untranslated"),
    "intron/pre-mRNA": ("intron", "pre-mrna", "pre‐mrna", "splice"),
}


def _load_openalex_search(script_path: str):
    spec = importlib.util.spec_from_file_location("nature_academic_search", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OpenAlex search helper: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.search


def _extract_pdf_rbps(pdf_path: str) -> list[str]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        start = lines.index("Motif contribution to full-CDS mean signal, Δlog2 (TE)") + 1
    except ValueError:
        start = next(i + 1 for i, line in enumerate(lines) if "full-CDS mean signal" in line)
    end = next(
        i for i in range(start, len(lines))
        if lines[i].startswith("Independently mutated RBP motifs")
    )
    excluded = {
        "Motif region", "5′UTR", "CDS", "3′UTR", "Motif effect",
        "Positive effect", "Negative effect", "FDR > 0.05", "No. transcripts",
    }
    rbps = []
    for label in lines[start:end]:
        if label in excluded or re.fullmatch(r"[−\-+]?\d[\d,.]*", label):
            continue
        clean = label.strip()
        if clean and clean not in rbps:
            rbps.append(clean)
    return rbps


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def _symbol_pattern(symbol: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", re.I)


def _classify_paper(symbol: str, paper: dict[str, Any]) -> dict[str, Any] | None:
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    text = f"{title}. {abstract}"
    symbol_re = _symbol_pattern(symbol)
    if not symbol_re.search(text):
        return None

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    context_sentences = []
    for index, sentence in enumerate(sentences):
        if not symbol_re.search(sentence):
            continue
        context_sentences.append(sentence)
        if index + 1 < len(sentences):
            next_sentence = sentences[index + 1]
            if re.match(
                r"^(it|this (?:protein|factor|rbp)|we|our results|these results)\b",
                next_sentence,
                flags=re.I,
            ):
                context_sentences.append(next_sentence)
    evidence_text = " ".join(dict.fromkeys(context_sentences))

    mechanisms = []
    if _contains_any(evidence_text, TRANSLATION_TERMS):
        mechanisms.append("direct_translation")
    if _contains_any(evidence_text, STABILITY_TERMS):
        mechanisms.append("stability_or_decay")
    if _contains_any(evidence_text, NUCLEAR_TERMS):
        mechanisms.append("nuclear_export_or_retention")
    if _contains_any(evidence_text, SPLICING_TERMS):
        mechanisms.append("pre_mRNA_splicing")
    if _contains_any(evidence_text, LNCRNA_TERMS):
        mechanisms.append("lncRNA")
    if not mechanisms:
        mechanisms.append("general_RNA_biology")

    if "direct_translation" in mechanisms:
        level = "A"
    elif "stability_or_decay" in mechanisms or "nuclear_export_or_retention" in mechanisms:
        level = "B"
    elif "pre_mRNA_splicing" in mechanisms:
        level = "C"
    else:
        level = "D"

    regions = [
        region for region, terms in REGION_PATTERNS.items()
        if _contains_any(evidence_text, terms)
    ]
    has_positive = _contains_any(evidence_text, POSITIVE_TERMS)
    has_negative = _contains_any(evidence_text, NEGATIVE_TERMS)
    if has_positive and has_negative:
        direction = "context_dependent_or_mixed"
    elif has_positive:
        direction = "positive_candidate"
    elif has_negative:
        direction = "negative_candidate"
    else:
        direction = "not_resolved_from_abstract"

    level_score = {"A": 40, "B": 30, "C": 20, "D": 5}[level]
    title_bonus = 15 if symbol_re.search(title) else 0
    citations = int(paper.get("cited_by_count") or 0)
    rank_score = level_score + title_bonus + min(math.log1p(citations), 8)
    return {
        "RBP_Name": symbol,
        "Evidence_Level": level,
        "Mechanisms": ";".join(mechanisms),
        "Literature_Regions": ";".join(regions) if regions else "not_reported",
        "Direction_Screen": direction,
        "Title": title,
        "Year": paper.get("year"),
        "Journal": paper.get("journal") or "",
        "DOI": paper.get("doi") or "",
        "PMID": paper.get("pmid") or "",
        "OpenAlex_ID": paper.get("openalex_id") or "",
        "Cited_By_Count": citations,
        "Evidence_Context": evidence_text,
        "Abstract": abstract,
        "Audit_Rank_Score": rank_score,
    }


def _search_europepmc(query: str, limit: int) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": str(limit),
    }
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
        + urllib.parse.urlencode(params)
    )
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "TRACE-RBP-literature-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    papers = []
    for result in payload.get("resultList", {}).get("result", []):
        papers.append({
            "title": result.get("title") or "",
            "doi": result.get("doi") or "",
            "pmid": result.get("pmid") or "",
            "authors": [result.get("authorString") or ""],
            "year": result.get("pubYear"),
            "publication_date": result.get("firstPublicationDate") or "",
            "cited_by_count": result.get("citedByCount") or 0,
            "journal": result.get("journalTitle") or "",
            "abstract": result.get("abstractText") or "",
            "openalex_id": "",
        })
    return papers


def _query_one(
    symbol: str,
    search_fn,
    cache_dir: Path,
    limit: int,
    request_delay: float,
    max_retries: int,
    backend: str,
    europepmc_sort_cited: bool,
) -> tuple[str, list[dict[str, Any]], str]:
    cache_path = cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', symbol)}.json"
    if cache_path.exists():
        return symbol, json.loads(cache_path.read_text()), "cache"

    if backend == "europepmc":
        mechanism_query = (
            "translation OR translational OR ribosome OR splicing OR spliceosome OR "
            '"pre-mRNA" OR "intron retention" OR "nuclear export" OR '
            '"nuclear retention" OR "RNA stability" OR "RNA decay"'
        )
        query = f'TITLE_ABS:"{symbol}" AND ({mechanism_query}) AND SRC:MED'
        if europepmc_sort_cited:
            query += " sort_cited:y"
    else:
        query = (
            f'"{symbol}" RNA binding protein translation translational ribosome '
            "splicing pre-mRNA intron retention nuclear export nuclear retention "
            "RNA stability RNA decay"
        )
    papers = None
    for attempt in range(max_retries + 1):
        try:
            if request_delay > 0:
                time.sleep(request_delay)
            if backend == "europepmc":
                papers = _search_europepmc(query=query, limit=limit)
            else:
                papers = search_fn(query=query, limit=limit, sort="relevance_score")
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt >= max_retries:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                wait_seconds = min(float(retry_after), 60.0)
            except (TypeError, ValueError):
                wait_seconds = min(5.0 * (2 ** attempt), 60.0)
            print(
                f"OpenAlex rate limit for {symbol}; retrying in {wait_seconds:.1f}s ",
                f"({attempt + 1}/{max_retries}).",
                flush=True,
            )
            time.sleep(wait_seconds)
    if papers is None:
        raise RuntimeError(f"No search result returned for {symbol}.")
    cache_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2))
    return symbol, papers, "network"


def _summarize_model_effects(summary: pd.DataFrame, rbps: list[str]) -> pd.DataFrame:
    working = summary.copy()
    working["RBP_Name"] = working["RBP_Name"].astype(str).str.strip()
    working = working[working["RBP_Name"].isin(rbps)].copy()
    for col in ("Median_Delta_Log2_TE", "FDR_BH", "N_Transcripts"):
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working["Abs_Effect"] = working["Median_Delta_Log2_TE"].abs()
    strongest = (
        working.sort_values(
            ["RBP_Name", "Abs_Effect", "FDR_BH"],
            ascending=[True, False, True],
        )
        .groupby("RBP_Name", as_index=False)
        .first()
    )
    regional = (
        working.sort_values(["RBP_Name", "Region"])
        .groupby("RBP_Name")
        .apply(
            lambda x: " | ".join(
                f"{r.Region}:{r.Median_Delta_Log2_TE:+.5f} (FDR={r.FDR_BH:.2g}, n={int(r.N_Transcripts)})"
                for r in x.itertuples()
            ),
            include_groups=False,
        )
        .rename("All_Model_Region_Effects")
        .reset_index()
    )
    cols = [
        "RBP_Name", "Region", "Median_Delta_Log2_TE", "FDR_BH",
        "N_Transcripts", "Direction",
    ]
    strongest = strongest[cols].rename(
        columns={
            "Region": "Strongest_Model_Region",
            "Median_Delta_Log2_TE": "Strongest_Model_Delta_Log2_TE",
            "FDR_BH": "Strongest_Model_FDR",
            "N_Transcripts": "Strongest_Model_N_Transcripts",
            "Direction": "Strongest_Model_Direction",
        }
    )
    return pd.DataFrame({"RBP_Name": rbps}).merge(strongest, how="left").merge(regional, how="left")


def _aggregate_literature(model: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in model.itertuples(index=False):
        subset = evidence[evidence["RBP_Name"] == row.RBP_Name].copy()
        if subset.empty:
            records.append({
                **row._asdict(),
                "Best_Evidence_Level": "None",
                "Evidence_Mechanisms": "no_symbol-specific_hit",
                "Literature_Direction_Screen": "not_available",
                "Concordance_Screen": "not_assessable",
                "Top_References": "",
                "N_Relevant_Papers_Screened": 0,
            })
            continue
        subset = subset.sort_values("Audit_Rank_Score", ascending=False)
        best_level = min(subset["Evidence_Level"], key=lambda x: "ABCD".index(x))
        mechanisms = sorted({m for value in subset["Mechanisms"] for m in value.split(";")})
        directions = set(subset["Direction_Screen"])
        if "context_dependent_or_mixed" in directions or (
            "positive_candidate" in directions and "negative_candidate" in directions
        ):
            lit_direction = "context_dependent_or_mixed"
        elif "positive_candidate" in directions:
            lit_direction = "positive_candidate"
        elif "negative_candidate" in directions:
            lit_direction = "negative_candidate"
        else:
            lit_direction = "not_resolved_from_abstract"

        model_direction = str(row.Strongest_Model_Direction).lower()
        if lit_direction.startswith("positive") and "positive" in model_direction:
            concordance = "candidate_concordant"
        elif lit_direction.startswith("negative") and "negative" in model_direction:
            concordance = "candidate_concordant"
        elif lit_direction.startswith(("positive", "negative")):
            concordance = "candidate_discordant"
        elif lit_direction == "context_dependent_or_mixed":
            concordance = "context_dependent"
        else:
            concordance = "not_assessable"

        refs = []
        for paper in subset.head(3).itertuples(index=False):
            identifier = paper.DOI or (f"PMID:{paper.PMID}" if paper.PMID else paper.OpenAlex_ID)
            refs.append(f"{paper.Title} ({paper.Year}; {identifier})")
        records.append({
            **row._asdict(),
            "Best_Evidence_Level": best_level,
            "Evidence_Mechanisms": ";".join(mechanisms),
            "Literature_Direction_Screen": lit_direction,
            "Concordance_Screen": concordance,
            "Top_References": " || ".join(refs),
            "N_Relevant_Papers_Screened": len(subset),
        })
    return pd.DataFrame(records)


def _write_markdown(audit: pd.DataFrame, output_path: Path) -> None:
    counts = audit["Best_Evidence_Level"].value_counts().to_dict()
    lines = [
        "# RBP literature audit",
        "",
        "## Scope and interpretation",
        "",
        f"The audit covers all {len(audit)} RBP labels extracted from the supplied significant-effect PDF.",
        "Candidate evidence levels are assigned from symbol-proximal title/abstract text: A, translation/ribosome; B, RNA stability/decay or nuclear transport; C, pre-mRNA splicing/intron retention; D, general RNA biology only; None, no symbol-specific hit in the screened records.",
        "Direction labels are conservative abstract-level screening annotations and require full-text verification before manuscript use.",
        "Splicing evidence supports an RNA-maturation hypothesis but does not by itself prove nuclear retention or reduced translation in this dataset.",
        "",
        "## Evidence-level counts",
        "",
        *[f"- {level}: {counts.get(level, 0)}" for level in ("A", "B", "C", "D", "None")],
        "",
        "## Complete RBP audit",
        "",
        "| RBP | strongest model region | Δlog2(TE) | FDR | evidence | mechanisms | direction screen | concordance screen |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in audit.itertuples(index=False):
        delta = "NA" if pd.isna(row.Strongest_Model_Delta_Log2_TE) else f"{row.Strongest_Model_Delta_Log2_TE:+.5f}"
        fdr = "NA" if pd.isna(row.Strongest_Model_FDR) else f"{row.Strongest_Model_FDR:.2g}"
        lines.append(
            f"| {row.RBP_Name} | {row.Strongest_Model_Region} | {delta} | {fdr} | "
            f"{row.Best_Evidence_Level} | {row.Evidence_Mechanisms} | "
            f"{row.Literature_Direction_Screen} | {row.Concordance_Screen} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--significant-pdf", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--search-helper",
        default="/Users/chunfu/.codex/skills/nature-academic-search/scripts/academic_search.py",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backend", choices=("europepmc", "openalex"), default="europepmc")
    parser.add_argument("--europepmc-sort-cited", action="store_true")
    parser.add_argument("--merge-europepmc-relevance-cache", action="store_true")
    parser.add_argument("--papers-per-rbp", type=int, default=12)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_suffix = "_cited" if args.backend == "europepmc" and args.europepmc_sort_cited else ""
    cache_dir = out_dir / f"{args.backend}{cache_suffix}_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rbps = _extract_pdf_rbps(args.significant_pdf)
    model = _summarize_model_effects(pd.read_csv(args.summary_csv), rbps)
    search_fn = None
    if args.backend == "openalex":
        search_fn = _load_openalex_search(args.search_helper)

    results: dict[str, list[dict[str, Any]]] = {}
    completed = 0
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _query_one,
                rbp,
                search_fn,
                cache_dir,
                args.papers_per_rbp,
                args.request_delay,
                args.max_retries,
                args.backend,
                args.europepmc_sort_cited,
            ): rbp
            for rbp in rbps
        }
        for future in concurrent.futures.as_completed(futures):
            rbp = futures[future]
            try:
                symbol, papers, source = future.result()
                results[symbol] = papers
            except Exception as exc:
                results[rbp] = []
                source = f"error:{exc}"
            with lock:
                completed += 1
                print(f"[{completed:03d}/{len(rbps):03d}] {rbp}: {len(results[rbp])} records ({source})", flush=True)

    if args.merge_europepmc_relevance_cache:
        relevance_cache = out_dir / "europepmc_cache"
        for rbp in rbps:
            path = relevance_cache / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', rbp)}.json"
            if not path.exists():
                continue
            merged = list(results.get(rbp, [])) + json.loads(path.read_text())
            unique = {}
            for paper in merged:
                key = (
                    str(paper.get("doi") or "").lower()
                    or str(paper.get("pmid") or "")
                    or str(paper.get("title") or "").lower()
                )
                unique.setdefault(key, paper)
            results[rbp] = list(unique.values())

    evidence_records = []
    for rbp in rbps:
        for paper in results.get(rbp, []):
            record = _classify_paper(rbp, paper)
            if record is not None:
                evidence_records.append(record)
    evidence = pd.DataFrame(evidence_records)
    if evidence.empty:
        evidence = pd.DataFrame(columns=[
            "RBP_Name", "Evidence_Level", "Mechanisms", "Literature_Regions",
            "Direction_Screen", "Title", "Year", "Journal", "DOI", "PMID",
            "OpenAlex_ID", "Cited_By_Count", "Evidence_Context", "Abstract",
            "Audit_Rank_Score",
        ])
    else:
        evidence = evidence.sort_values(
            ["RBP_Name", "Audit_Rank_Score"], ascending=[True, False]
        )
    audit = _aggregate_literature(model, evidence)

    evidence.to_csv(out_dir / "rbp_literature_evidence_long.csv", index=False)
    audit.to_csv(out_dir / "rbp_literature_audit.csv", index=False)
    _write_markdown(audit, out_dir / "rbp_literature_audit.md")
    (out_dir / "rbp_labels_from_pdf.txt").write_text("\n".join(rbps) + "\n")
    print(f"Wrote audit for {len(rbps)} RBPs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
