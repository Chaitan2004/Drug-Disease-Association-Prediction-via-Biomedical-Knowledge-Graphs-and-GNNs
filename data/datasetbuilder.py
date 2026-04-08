import pandas as pd

cols = [
    "ChemicalName",
    "ChemicalID",
    "CASRN",
    "DiseaseName",
    "DiseaseID",
    "DirectEvidence",
    "PMID"
]

ctd = pd.read_csv(
    "CTD_curated_chemicals_diseases.csv",   # <-- your filename
    names=cols,
    header=None
)


print(ctd.columns)

ctd_ther = ctd[ctd["DirectEvidence"] == "therapeutic"]

print("Therapeutic associations:", len(ctd_ther))

drug_df = (
    ctd_ther[["ChemicalID", "ChemicalName"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

drug_df["drug_index"] = drug_df.index

disease_df = (
    ctd_ther[["DiseaseID", "DiseaseName"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

disease_df["disease_index"] = disease_df.index

import numpy as np

num_drugs = len(drug_df)
num_diseases = len(disease_df)

ther_matrix = np.zeros((num_drugs, num_diseases), dtype=int)

drug_id_to_idx = dict(
    zip(drug_df["ChemicalID"], drug_df["drug_index"])
)
disease_id_to_idx = dict(
    zip(disease_df["DiseaseID"], disease_df["disease_index"])
)

for _, row in ctd_ther.iterrows():
    i = drug_id_to_idx[row["ChemicalID"]]
    j = disease_id_to_idx[row["DiseaseID"]]
    ther_matrix[i, j] = 1

np.savetxt("therapeutic2.txt", ther_matrix)
np.savetxt("drug_disease2.csv", ther_matrix, delimiter=",")

drug_df.to_csv(
    "drug_vocab.csv",
    index=False
)

disease_df.to_csv(
    "disease_vocab.csv",
    index=False
)

# Drug-drug similarity using Jaccard index over therapeutic associations
# Jaccard(i, j) = |A ∩ B| / |A ∪ B| where A,B are sets of diseases per drug.
row_sums = ther_matrix.sum(axis=1).astype(float)
intersection = ther_matrix @ ther_matrix.T
union = row_sums[:, None] + row_sums[None, :] - intersection

jaccard = np.zeros_like(intersection, dtype=float)
nonzero_union = union != 0
jaccard[nonzero_union] = intersection[nonzero_union] / union[nonzero_union]

np.savetxt("drug_sim2.csv", jaccard, delimiter=",")

# Disease-disease similarity using MeSH DAG-based semantic similarity
# as described in HDGAT (Delta = 0.5)
import os
from pathlib import Path

DELTA = 0.5


def resolve_local_path(filename):
    candidate = Path(filename)
    if candidate.exists():
        return candidate
    candidate = Path("data") / filename
    if candidate.exists():
        return candidate
    return None


def find_mesh_tree_file():
    candidates = [
        "mesh_tree_numbers.csv",
        "mesh_tree_numbers.tsv",
        "MeSH_tree_numbers.csv",
        "MeSH_tree_numbers.tsv",
        "disease_mesh_tree_numbers.csv",
        "disease_mesh_tree_numbers.tsv",
        "mesh_treenumbers.csv",
        "mesh_treenumbers.tsv",
    ]
    for name in candidates:
        path = resolve_local_path(name)
        if path is not None:
            return path
    return None


def find_mesh_descriptor_xml():
    # Common MeSH descriptor record filenames (year varies)
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
        path = resolve_local_path(name)
        if path is not None:
            return path
    # Fallback: any desc*.xml in data/
    data_dir = Path("data")
    if data_dir.exists():
        matches = sorted(data_dir.glob("desc*.xml"))
        if matches:
            return matches[0]
    return None


def load_mesh_descriptor_xml(mesh_path):
    """
    Parse MeSH DescriptorRecordSet XML and return dict:
    { 'MESH:Dxxxxxx': [tree_number, ...], ... }
    """
    import xml.etree.ElementTree as ET

    mesh_map = {}
    for _, elem in ET.iterparse(str(mesh_path), events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue

        descriptor_ui = elem.findtext("DescriptorUI")
        if not descriptor_ui:
            elem.clear()
            continue

        tree_numbers = [
            tn.text
            for tn in elem.findall("./TreeNumberList/TreeNumber")
            if tn is not None and tn.text
        ]
        if tree_numbers:
            mesh_map[f"MESH:{descriptor_ui}"] = tree_numbers
        elem.clear()

    return mesh_map


def load_mesh_tree_numbers():
    mesh_path = find_mesh_tree_file()
    if mesh_path is None:
        xml_path = find_mesh_descriptor_xml()
        if xml_path is None:
            raise FileNotFoundError(
                "Could not find a MeSH tree-number file or descriptor XML. Please "
                "provide either a file with columns like 'DiseaseID' and "
                "'TreeNumber(s)' or a MeSH descriptor XML file (e.g., desc2026.xml) "
                "in the data/ folder."
            )
        return load_mesh_descriptor_xml(xml_path)

    sep = "\t" if str(mesh_path).endswith(".tsv") else ","
    mesh_df = pd.read_csv(mesh_path, sep=sep)

    # Normalize column names
    cols_lower = {c.lower(): c for c in mesh_df.columns}
    if "diseaseid" in cols_lower:
        disease_col = cols_lower["diseaseid"]
    elif "mesh_id" in cols_lower:
        disease_col = cols_lower["mesh_id"]
    else:
        raise ValueError(
            "MeSH tree-number file must include a 'DiseaseID' (or 'mesh_id') column."
        )

    if "treenumber" in cols_lower:
        tree_col = cols_lower["treenumber"]
        mesh_df[tree_col] = mesh_df[tree_col].astype(str)
        grouped = mesh_df.groupby(disease_col)[tree_col].apply(list)
    elif "treenumbers" in cols_lower:
        tree_col = cols_lower["treenumbers"]
        # Allow delimiter-separated list per disease
        def split_numbers(val):
            if pd.isna(val):
                return []
            s = str(val)
            for delim in [";", "|", ",", " "]:
                if delim in s:
                    parts = [p for p in s.split(delim) if p]
                    return parts
            return [s]
        grouped = mesh_df.groupby(disease_col)[tree_col].apply(
            lambda vals: [tn for v in vals for tn in split_numbers(v)]
        )
    else:
        raise ValueError(
            "MeSH tree-number file must include 'TreeNumber' or 'TreeNumbers' column."
        )

    # Map DiseaseID -> list of tree numbers
    return grouped.to_dict()


def build_dag_contributions(tree_numbers):
    """
    Build contribution map WA(d) for a disease from its MeSH tree numbers.
    Nodes are represented by tree-number strings. For multiple trees, take max.
    """
    contrib = {}
    for tn in tree_numbers:
        if not tn or tn == "nan":
            continue
        parts = tn.split(".")
        for depth in range(len(parts)):
            ancestor = ".".join(parts[: depth + 1])
            distance = len(parts) - depth - 1
            value = DELTA ** distance
            if value > contrib.get(ancestor, 0.0):
                contrib[ancestor] = value
    return contrib


mesh_tree_numbers = load_mesh_tree_numbers()

# Build semantic contributions for each disease in the vocab
contribs = []
dv = np.zeros(len(disease_df), dtype=float)
for idx, row in disease_df.iterrows():
    disease_id = row["DiseaseID"]
    tree_numbers = mesh_tree_numbers.get(disease_id, [])
    contrib = build_dag_contributions(tree_numbers)
    contribs.append(contrib)
    dv[idx] = sum(contrib.values())

# Compute disease-disease similarity matrix
num_diseases = len(disease_df)
dis_sim = np.zeros((num_diseases, num_diseases), dtype=float)
for i in range(num_diseases):
    dis_sim[i, i] = 1.0 if dv[i] > 0 else 0.0
    keys_i = set(contribs[i].keys())
    for j in range(i + 1, num_diseases):
        if dv[i] == 0 or dv[j] == 0:
            sim = 0.0
        else:
            keys_j = set(contribs[j].keys())
            common = keys_i & keys_j
            if not common:
                sim = 0.0
            else:
                numerator = sum(contribs[i][d] + contribs[j][d] for d in common)
                sim = numerator / (dv[i] + dv[j])
        dis_sim[i, j] = sim
        dis_sim[j, i] = sim

np.savetxt("disease_sim2.csv", dis_sim, delimiter=",")
