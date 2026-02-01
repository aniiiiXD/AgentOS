"""Stage: featurize molecules using RDKit molecular fingerprints."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ensure_sdk_on_path, log_line, maybe_fail_once, read_json, wait_for_message, write_json

ensure_sdk_on_path()
from clove_sdk import CloveClient  # noqa: E402

STAGE_NAME = "featurize"


def compute_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> Optional[List[int]]:
    """Compute Morgan (circular) fingerprint for a molecule.

    Args:
        smiles: SMILES string representation of molecule
        radius: Radius for Morgan fingerprint (default: 2, equivalent to ECFP4)
        n_bits: Number of bits in fingerprint vector

    Returns:
        List of integers (0/1) representing the fingerprint, or None if invalid SMILES
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(fp)
    except Exception:
        return None


def compute_rdkit_fingerprint(smiles: str, n_bits: int = 1024) -> Optional[List[int]]:
    """Compute RDKit topological fingerprint.

    Args:
        smiles: SMILES string representation of molecule
        n_bits: Number of bits in fingerprint vector

    Returns:
        List of integers (0/1) representing the fingerprint, or None if invalid SMILES
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RDKFingerprint

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = RDKFingerprint(mol, fpSize=n_bits)
        return list(fp)
    except Exception:
        return None


def compute_maccs_keys(smiles: str) -> Optional[List[int]]:
    """Compute MACCS structural keys (166 bits).

    Args:
        smiles: SMILES string representation of molecule

    Returns:
        List of integers (0/1) representing MACCS keys, or None if invalid SMILES
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = MACCSkeys.GenMACCSKeys(mol)
        return list(fp)
    except Exception:
        return None


def compute_descriptors(smiles: str) -> Optional[dict]:
    """Compute molecular descriptors using RDKit.

    Args:
        smiles: SMILES string representation of molecule

    Returns:
        Dictionary of descriptor names and values, or None if invalid SMILES
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return {
            "mol_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "aromatic_rings": Lipinski.NumAromaticRings(mol),
            "heavy_atoms": Lipinski.HeavyAtomCount(mol),
            "fraction_sp3": Lipinski.FractionCSP3(mol),
        }
    except Exception:
        return None


def featurize_molecule(smiles: str, method: str = "morgan", **kwargs) -> Optional[dict]:
    """Featurize a single molecule.

    Args:
        smiles: SMILES string
        method: Featurization method ('morgan', 'rdkit', 'maccs', 'descriptors', 'combined')
        **kwargs: Additional arguments for fingerprint methods (radius, n_bits)

    Returns:
        Dictionary with feature vector and metadata
    """
    result = {"smiles": smiles, "valid": False}

    if method == "morgan":
        radius = kwargs.get("radius", 2)
        n_bits = kwargs.get("n_bits", 1024)
        fp = compute_morgan_fingerprint(smiles, radius=radius, n_bits=n_bits)
        if fp is not None:
            result["features"] = fp
            result["valid"] = True
            result["n_features"] = len(fp)
            result["method"] = f"morgan_r{radius}_{n_bits}bits"

    elif method == "rdkit":
        n_bits = kwargs.get("n_bits", 1024)
        fp = compute_rdkit_fingerprint(smiles, n_bits=n_bits)
        if fp is not None:
            result["features"] = fp
            result["valid"] = True
            result["n_features"] = len(fp)
            result["method"] = f"rdkit_{n_bits}bits"

    elif method == "maccs":
        fp = compute_maccs_keys(smiles)
        if fp is not None:
            result["features"] = fp
            result["valid"] = True
            result["n_features"] = len(fp)
            result["method"] = "maccs_166bits"

    elif method == "descriptors":
        desc = compute_descriptors(smiles)
        if desc is not None:
            result["features"] = list(desc.values())
            result["descriptor_names"] = list(desc.keys())
            result["valid"] = True
            result["n_features"] = len(desc)
            result["method"] = "rdkit_descriptors"

    elif method == "combined":
        # Combine Morgan fingerprint with descriptors
        radius = kwargs.get("radius", 2)
        n_bits = kwargs.get("n_bits", 512)  # Smaller FP when combining
        fp = compute_morgan_fingerprint(smiles, radius=radius, n_bits=n_bits)
        desc = compute_descriptors(smiles)

        if fp is not None and desc is not None:
            result["features"] = fp + list(desc.values())
            result["descriptor_names"] = [f"morgan_{i}" for i in range(len(fp))] + list(desc.keys())
            result["valid"] = True
            result["n_features"] = len(fp) + len(desc)
            result["method"] = f"combined_morgan{n_bits}_descriptors"

    return result


def main() -> int:
    client = CloveClient()
    if not client.connect():
        print("[featurize] ERROR: Failed to connect to Clove kernel")
        return 1

    try:
        client.register_name(STAGE_NAME)
        message = wait_for_message(client, expected_type="run_stage", expected_stage=STAGE_NAME)

        run_id = message.get("run_id", "run_000")
        artifacts_dir = Path(message.get("artifacts_dir", "artifacts"))
        logs_dir = Path(message.get("logs_dir", "logs"))
        config = message.get("config", {})
        input_payload = message.get("input", {})
        reply_to = message.get("reply_to", "orchestrator")

        run_dir = artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / run_id / f"{STAGE_NAME}.log"

        log_line(log_path, "stage start")
        maybe_fail_once(run_dir, STAGE_NAME, config)

        # Load dataset
        dataset_path = Path(input_payload.get("dataset_path", run_dir / "dataset.json"))
        splits_path = Path(input_payload.get("splits_path", run_dir / "splits.json"))

        dataset = read_json(dataset_path)
        molecules = dataset.get("molecules", [])
        splits_info = read_json(splits_path)

        # Get featurization config
        method = config.get("method", "morgan")
        radius = config.get("radius", 2)
        n_bits = config.get("n_bits", 1024)

        log_line(log_path, f"featurizing {len(molecules)} molecules with method={method}")

        # Featurize all molecules
        features = []
        valid_count = 0
        invalid_count = 0

        for mol in molecules:
            mol_id = mol.get("id")
            smiles = mol.get("smiles", "")
            label = mol.get("label")
            split = mol.get("split")

            feat_result = featurize_molecule(smiles, method=method, radius=radius, n_bits=n_bits)

            if feat_result and feat_result.get("valid"):
                features.append({
                    "id": mol_id,
                    "features": feat_result["features"],
                    "label": label,
                    "split": split,
                    "n_features": feat_result["n_features"],
                })
                valid_count += 1
            else:
                invalid_count += 1
                log_line(log_path, f"invalid SMILES for {mol_id}: {smiles[:50]}...")

        # Get feature dimension from first valid entry
        n_features = features[0]["n_features"] if features else 0

        features_path = run_dir / "features.json"
        write_json(features_path, {
            "features": features,
            "method": method,
            "n_features": n_features,
            "config": {"radius": radius, "n_bits": n_bits},
        })

        log_line(log_path, f"featurized {valid_count} molecules, {invalid_count} invalid")

        output = {
            "features_path": str(features_path),
            "dataset_path": str(dataset_path),
            "splits_path": str(splits_path),
            "count": len(features),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "n_features": n_features,
        }
        metadata = {
            "method": method,
            "radius": radius if method in ["morgan", "combined"] else None,
            "n_bits": n_bits if method in ["morgan", "rdkit", "combined"] else None,
        }
        stage_result = {
            "type": "stage_complete",
            "stage": STAGE_NAME,
            "run_id": run_id,
            "status": "ok",
            "output": output,
            "metadata": metadata,
        }

        write_json(run_dir / f"{STAGE_NAME}.json", stage_result)
        client.store(f"pipeline:{run_id}:{STAGE_NAME}", stage_result, scope="global")
        client.send_message(stage_result, to_name=reply_to)
        log_line(log_path, "stage complete")
    finally:
        client.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
