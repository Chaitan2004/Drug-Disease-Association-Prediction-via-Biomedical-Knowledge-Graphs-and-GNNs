import json
import time
import urllib.parse
import urllib.request
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

# Cache to avoid re-querying the same names
PUBCHEM_CACHE_PATH = Path("pubchem_cache.json")


def _load_cache() -> Dict[str, dict]:
    if PUBCHEM_CACHE_PATH.exists():
        try:
            return json.loads(PUBCHEM_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, dict]) -> None:
    PUBCHEM_CACHE_PATH.write_text(json.dumps(cache))


def _fetch_json(url: str) -> Optional[dict]:
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


def _extract_pubchem_text(name: str, cache: Dict[str, dict]) -> Tuple[str, Dict[str, dict]]:
    if name in cache:
        return cache[name].get("text", ""), cache

    # Basic retry loop
    for attempt in range(3):
        try:
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
            cache[name] = {"text": text}
            return text, cache
        except Exception:
            if attempt == 2:
                break
            time.sleep(PUBCHEM_RATE_SECONDS)

    # Fallback: minimal text
    text = f"Drug: {name}."
    cache[name] = {"text": text}
    return text, cache


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
    return mapping


def build_drug_texts(drug_vocab_path: str, out_path: str) -> None:
    drug_df = pd.read_csv(drug_vocab_path)
    cache = _load_cache()

    texts = []
    for name in drug_df["ChemicalName"].astype(str).tolist():
        text, cache = _extract_pubchem_text(name, cache)
        texts.append(text)

    _save_cache(cache)
    drug_df["Text"] = texts
    drug_df[["ChemicalID", "Text"]].to_csv(out_path, index=False)


def build_disease_texts(disease_vocab_path: str, out_path: str) -> None:
    disease_df = pd.read_csv(disease_vocab_path)
    mesh_xml = _find_mesh_descriptor_xml()
    if mesh_xml is None:
        raise FileNotFoundError(
            "MeSH descriptor XML not found. Please place desc20xx.xml in data/."
        )

    mesh_map = _load_mesh_scope_notes(mesh_xml)

    texts = []
    for mesh_id in disease_df["DiseaseID"].astype(str).tolist():
        ui = mesh_id.replace("MESH:", "")
        name, scope = mesh_map.get(ui, ("", ""))

        parts = [f"Disease: {name or mesh_id}."]
        parts.append(f"MeSH ID: {mesh_id}.")
        if scope:
            parts.append(f"ScopeNote: {scope}.")
        texts.append(" ".join(parts))

    disease_df["Text"] = texts
    disease_df[["DiseaseID", "Text"]].to_csv(out_path, index=False)


def main() -> None:
    # If small vocab files exist, prefer them to speed up text collection.
    drug_vocab = "drug_vocab_small.csv" if Path("drug_vocab_small.csv").exists() else "drug_vocab.csv"
    disease_vocab = "disease_vocab_small.csv" if Path("disease_vocab_small.csv").exists() else "disease_vocab.csv"

    drug_out = "drug_texts_small.csv" if drug_vocab.endswith("_small.csv") else "drug_texts.csv"
    disease_out = "disease_texts_small.csv" if disease_vocab.endswith("_small.csv") else "disease_texts.csv"

    build_drug_texts(drug_vocab, drug_out)
    build_disease_texts(disease_vocab, disease_out)


if __name__ == "__main__":
    main()
