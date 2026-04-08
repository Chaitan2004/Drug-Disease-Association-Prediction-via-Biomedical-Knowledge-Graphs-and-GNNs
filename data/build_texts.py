import argparse
import json
import logging
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# ---------------------------
# PubChem (PUG REST) settings
# ---------------------------
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_RATE_SECONDS = 0.25  # ~4 requests/sec (safer than 5/sec)
PUBCHEM_TIMEOUT = 30
PUBCHEM_VERBOSE_NOT_FOUND_LIMIT = 10
PUBCHEM_VERBOSE_FOUND_LIMIT = 10
PROGRESS_LOG_INTERVAL = 100
LOGGER = logging.getLogger(__name__)

# Cache to avoid re-querying the same names
PUBCHEM_CACHE_PATH = Path("pubchem_cache.json")


def _load_cache() -> Dict[str, dict]:
    if PUBCHEM_CACHE_PATH.exists():
        try:
            cache = json.loads(PUBCHEM_CACHE_PATH.read_text())
            LOGGER.debug("Loaded %d cached PubChem entries from %s", len(cache), PUBCHEM_CACHE_PATH)
            return cache
        except Exception:
            LOGGER.exception("Failed to load PubChem cache from %s", PUBCHEM_CACHE_PATH)
            return {}
    LOGGER.debug("PubChem cache file not found at %s; starting fresh", PUBCHEM_CACHE_PATH)
    return {}


def _save_cache(cache: Dict[str, dict]) -> None:
    PUBCHEM_CACHE_PATH.write_text(json.dumps(cache))
    LOGGER.debug("Saved %d PubChem cache entries to %s", len(cache), PUBCHEM_CACHE_PATH)


def _fetch_json(url: str) -> Optional[dict]:
    LOGGER.debug("Fetching URL: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "HDGAT-text-builder/1.0"})
    with urllib.request.urlopen(req, timeout=PUBCHEM_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pubchem_property_by_name(name: str) -> Optional[dict]:
    quoted = urllib.parse.quote(name)
    url = (
        f"{PUBCHEM_BASE}/compound/name/{quoted}/property/"
        "Title,IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
    )
    return _fetch_json(url)


def _pubchem_synonyms_by_name(name: str) -> Optional[dict]:
    quoted = urllib.parse.quote(name)
    url = f"{PUBCHEM_BASE}/compound/name/{quoted}/synonyms/JSON"
    return _fetch_json(url)


def _extract_pubchem_text(name: str, cache: Dict[str, dict]) -> Tuple[str, str, Dict[str, dict]]:
    if name in cache:
        LOGGER.debug("PubChem cache hit for '%s'", name)
        return cache[name].get("text", ""), cache[name].get("status", "cached"), cache

    # Basic retry loop
    for attempt in range(3):
        try:
            LOGGER.debug("Fetching PubChem data for '%s' (attempt %d/3)", name, attempt + 1)
            prop = _pubchem_property_by_name(name)
            time.sleep(PUBCHEM_RATE_SECONDS)
            syn = _pubchem_synonyms_by_name(name)
            time.sleep(PUBCHEM_RATE_SECONDS)

            props = None
            if prop and "PropertyTable" in prop and prop["PropertyTable"].get("Properties"):
                props = prop["PropertyTable"]["Properties"][0]

            synonyms = []
            if syn and "InformationList" in syn and syn["InformationList"].get("Information"):
                synonyms = syn["InformationList"]["Information"][0].get("Synonym", [])[:10]

            parts = [f"Drug: {name}."]
            if props:
                title = props.get("Title")
                if title and title.lower() != name.lower():
                    parts.append(f"Title: {title}.")
                if props.get("IUPACName"):
                    parts.append(f"IUPAC: {props['IUPACName']}.")
                if props.get("MolecularFormula"):
                    parts.append(f"Formula: {props['MolecularFormula']}.")
                if props.get("MolecularWeight"):
                    parts.append(f"MolWt: {props['MolecularWeight']}.")
                if props.get("CanonicalSMILES"):
                    parts.append(f"SMILES: {props['CanonicalSMILES']}.")

            if synonyms:
                parts.append("Synonyms: " + "; ".join(synonyms) + ".")

            text = " ".join(parts)
            cache[name] = {"text": text, "status": "found"}
            LOGGER.debug(
                "Built PubChem text for '%s' with props=%s synonyms=%d",
                name,
                bool(props),
                len(synonyms),
            )
            return text, "found", cache
        except HTTPError as exc:
            if exc.code == 404:
                LOGGER.debug("PubChem not found for '%s'; using fallback text", name)
                break
            LOGGER.warning(
                "PubChem HTTP error for '%s' on attempt %d/3: %s",
                name,
                attempt + 1,
                exc,
            )
        except URLError as exc:
            LOGGER.warning(
                "PubChem network error for '%s' on attempt %d/3: %s",
                name,
                attempt + 1,
                exc,
            )
        except Exception:
            LOGGER.exception("Unexpected PubChem lookup failure for '%s' on attempt %d/3", name, attempt + 1)
            if attempt == 2:
                break
        if attempt < 2:
            time.sleep(PUBCHEM_RATE_SECONDS)

    # Fallback: minimal text
    text = f"Drug: {name}."
    cache[name] = {"text": text, "status": "fallback"}
    LOGGER.debug("Falling back to minimal PubChem text for '%s'", name)
    return text, "fallback", cache


# ---------------------------
# MeSH Descriptor XML parsing
# ---------------------------
def _find_mesh_descriptor_xml() -> Optional[Path]:
    candidates = [
        "desc2026.xml",
        "desc2025.xml",
        "desc2024.xml",
        "desc2023.xml",
        "desc2022.xml",
        "desc2021.xml",
        "desc2020.xml",
        "desc.xml",
    ]
    for name in candidates:
        p = Path(name)
        if p.exists():
            return p
        p = Path("data") / name
        if p.exists():
            return p
    # fallback: any desc*.xml in data/
    for p in Path("data").glob("desc*.xml"):
        return p
    return None


def _load_mesh_scope_notes(mesh_xml: Path) -> Dict[str, Tuple[str, str]]:
    """
    Return mapping: DescriptorUI -> (DescriptorName, ScopeNote)
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    for _, elem in ET.iterparse(str(mesh_xml), events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue

        descriptor_ui = elem.findtext("DescriptorUI")
        descriptor_name = elem.findtext("DescriptorName/String") or ""

        # ScopeNote usually lives under preferred Concept
        scope_note = ""
        concept_list = elem.findall("ConceptList/Concept")
        for concept in concept_list:
            if concept.attrib.get("PreferredConceptYN") == "Y":
                scope_note = concept.findtext("ScopeNote") or ""
                break

        if descriptor_ui:
            mapping[descriptor_ui] = (descriptor_name, scope_note)
        elem.clear()
    LOGGER.debug("Loaded %d MeSH descriptor records from %s", len(mapping), mesh_xml)
    return mapping


def build_drug_texts(drug_vocab_path: str, out_path: str) -> None:
    drug_df = pd.read_csv(drug_vocab_path)
    LOGGER.info("Building drug texts from %s (%d rows)", drug_vocab_path, len(drug_df))
    cache = _load_cache()

    texts = []
    found_count = 0
    fallback_count = 0
    for idx, name in enumerate(drug_df["ChemicalName"].astype(str).tolist(), start=1):
        text, status, cache = _extract_pubchem_text(name, cache)
        texts.append(text)
        if status == "found":
            found_count += 1
            if found_count <= PUBCHEM_VERBOSE_FOUND_LIMIT:
                LOGGER.info("PubChem found for '%s'", name)
            elif found_count == PUBCHEM_VERBOSE_FOUND_LIMIT + 1:
                LOGGER.info(
                    "Additional PubChem hits will be summarized only; rerun with --debug for full details"
                )
        else:
            fallback_count += 1
            if fallback_count <= PUBCHEM_VERBOSE_NOT_FOUND_LIMIT:
                LOGGER.warning("PubChem not found for '%s'; using fallback text", name)
            elif fallback_count == PUBCHEM_VERBOSE_NOT_FOUND_LIMIT + 1:
                LOGGER.warning(
                    "Additional PubChem misses will be summarized only; rerun with --debug for full details"
                )
        if idx % PROGRESS_LOG_INTERVAL == 0 or idx == len(drug_df):
            LOGGER.info(
                "Drug progress: %d/%d processed (%d PubChem hits, %d fallbacks)",
                idx,
                len(drug_df),
                found_count,
                fallback_count,
            )
        LOGGER.debug("Processed drug %d/%d: %s", idx, len(drug_df), name)

    _save_cache(cache)
    drug_df["Text"] = texts
    drug_df[["ChemicalID", "Text"]].to_csv(out_path, index=False)
    LOGGER.info(
        "Drug text summary: %d PubChem hits, %d fallback texts",
        found_count,
        fallback_count,
    )
    LOGGER.info("Wrote drug texts to %s", out_path)


def build_disease_texts(disease_vocab_path: str, out_path: str) -> None:
    disease_df = pd.read_csv(disease_vocab_path)
    LOGGER.info("Building disease texts from %s (%d rows)", disease_vocab_path, len(disease_df))
    mesh_xml = _find_mesh_descriptor_xml()
    if mesh_xml is None:
        raise FileNotFoundError(
            "MeSH descriptor XML not found. Please place desc20xx.xml in data/."
        )
    LOGGER.debug("Using MeSH descriptor XML: %s", mesh_xml)

    mesh_map = _load_mesh_scope_notes(mesh_xml)

    texts = []
    for idx, mesh_id in enumerate(disease_df["DiseaseID"].astype(str).tolist(), start=1):
        ui = mesh_id.replace("MESH:", "")
        name, scope = mesh_map.get(ui, ("", ""))

        parts = [f"Disease: {name or mesh_id}."]
        parts.append(f"MeSH ID: {mesh_id}.")
        if scope:
            parts.append(f"ScopeNote: {scope}.")
        else:
            LOGGER.debug("No MeSH scope note found for %s", mesh_id)
        texts.append(" ".join(parts))
        if idx % PROGRESS_LOG_INTERVAL == 0 or idx == len(disease_df):
            LOGGER.info("Disease progress: %d/%d processed", idx, len(disease_df))
        LOGGER.debug("Processed disease %d/%d: %s", idx, len(disease_df), mesh_id)

    disease_df["Text"] = texts
    disease_df[["DiseaseID", "Text"]].to_csv(out_path, index=False)
    LOGGER.info("Wrote disease texts to %s", out_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build drug and disease text files.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args()


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    # If small vocab files exist, prefer them to speed up text collection.
    drug_vocab = "drug_vocab_small.csv" if Path("drug_vocab_small.csv").exists() else "drug_vocab.csv"
    disease_vocab = "disease_vocab_small.csv" if Path("disease_vocab_small.csv").exists() else "disease_vocab.csv"

    drug_out = "drug_texts_small.csv" if drug_vocab.endswith("_small.csv") else "drug_texts.csv"
    disease_out = "disease_texts_small.csv" if disease_vocab.endswith("_small.csv") else "disease_texts.csv"

    LOGGER.info("Selected vocab files: drug=%s disease=%s", drug_vocab, disease_vocab)
    build_drug_texts(drug_vocab, drug_out)
    build_disease_texts(disease_vocab, disease_out)


if __name__ == "__main__":
    main()
