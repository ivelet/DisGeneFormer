"""
DiseaseNet.py — Heterogeneous disease-phenotype graph.

Changes from the original homogeneous version
----------------------------------------------
* HPO terms become first-class nodes ('hpo' node type) rather than flat binary
  features.  Phenotype information now flows through graph edges; HPO binary
  features are removed from the disease feature vector.
* Information Content (IC) is computed per HPO term using the true-path
  annotation propagation rule, making rare shared phenotypes worth far more
  than common ancestor terms.
* Four edge types are built:
    ('disease', 'annotated_with', 'hpo')   — IC-weighted direct annotation
    ('hpo',     'rev_annotated',  'disease') — reverse of the above
    ('hpo',     'is_a',           'hpo')   — ontology hierarchy from hp.obo
    ('disease', 'pheno_sim',      'disease') — Resnik/IC-weighted similarity
* Returns a PyG HeteroData object instead of a homogeneous Data object.

Required raw files (place in data/disease_net/raw/)
----------------------------------------------------
    hp.obo           Human Phenotype Ontology  (https://github.com/obophenotype/
                      human-phenotype-ontology/releases — download hp.obo)
    phenotype.hpoa   Disease–HPO annotations   (https://hpo.jax.org/app/data/
                      annotations — "Annotation information" → phenotype.hpoa)

New config keys (all have sensible defaults)
--------------------------------------------
    cfg.disease_net.obo_path                 default: data/disease_net/raw/hp.obo
    cfg.disease_net.hpoa_path                default: data/disease_net/raw/phenotype.hpoa
    cfg.disease_net.min_ic_for_disease_edge  default: 1.5   (filters out very
                                             common / uninformative shared terms
                                             when building disease–disease edges)

IMPORTANT — delete the old `processed/` directory before the first run so that
PyG triggers re-processing and builds the heterogeneous graph from scratch.
"""

import ast
import collections
import logging
import math
import os
import os.path as osp

from collections import defaultdict
from shutil import copyfile

import pandas as pd
import torch

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import InMemoryDataset, HeteroData


log = logging.getLogger(__name__)


class DiseaseNet(InMemoryDataset):
    """
    Heterogeneous disease-phenotype graph dataset.

    Node types
    ----------
    'disease' : one node per OMIM disease
                features = pathway one-hot + publication TF-IDF
                           (HPO binary features removed — info flows via edges)
    'hpo'     : one node per HPO term that directly annotates ≥1 disease
                features = [IC_norm, ontology_depth_norm, log_annotation_freq_norm]

    Edge types
    ----------
    ('disease', 'annotated_with', 'hpo')    IC-weighted direct annotation
    ('hpo',     'rev_annotated',  'disease') reverse of above
    ('hpo',     'is_a',           'hpo')    is-a ontology edges (child → parent)
    ('disease', 'pheno_sim',      'disease') IC-weighted Resnik similarity
    """

    # ── Enums ────────────────────────────────────────────────────────────────

    class RawFileEnum:
        disease_hpo                = 0
        disease_publication_titles = 1
        disease_pathway            = 2
        all_diseases               = 3
        hpoa                       = 4
        obo                        = 5

    class ProcessedFileEnum:
        disease_id_feature_index_mapping = 0
        hpo_id_index_mapping             = 1
        disease_features                 = 2
        hpo_features                     = 3
        edges                            = 4
        data                             = 5

    # ── Initialisation ───────────────────────────────────────────────────────

    def __init__(self, cfg, experiment_dir, transform=None, pre_transform=None):
        self.root      = os.path.join(experiment_dir, "disease_net")
        os.makedirs(self.root, exist_ok=True)
        self._raw_root = os.path.join("data/disease_net", "raw")
        os.makedirs(self._raw_root, exist_ok=True)

        dn = cfg.disease_net
        self.skip_truncated_svd     = dn.skip_truncated_svd
        self.svd_components         = dn.svd_components
        self.svd_n_iter             = dn.svd_n_iter
        self.edge_source            = dn.edge_source
        self.feature_source         = dn.features_to_use
        self.n_neighbours           = dn.n_neighbors
        self.train_positives_path   = dn.train_positives_path
        self.train_negatives_path   = dn.train_negatives_path
        self.gda_edges              = dn.gda_edges
        self.test_path              = cfg.evaluation.test_path
        self.umls_omim_mapping_path = cfg.evaluation.umls_omim_mapping_path

        # New keys — backward-compatible defaults so existing configs still work
        self.obo_path  = getattr(dn, 'obo_path',  'data/disease_net/raw/hp.obo')
        self.hpoa_path = getattr(dn, 'hpoa_path', 'data/disease_net/raw/phenotype.hpoa')
        self.min_ic_for_disease_edge = getattr(dn, 'min_ic_for_disease_edge', 1.5)

        self.seed = cfg.get('train', {}).get('seed', 42)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[self.ProcessedFileEnum.data]
        )

    # ── PyG file lists ───────────────────────────────────────────────────────

    @property
    def raw_dir(self):
        return str(self._raw_root)

    @property
    def raw_file_names(self):
        # Order matches RawFileEnum indices
        return [
            'disease_hpo.tsv',                               # 0
            'disease_publication_titles_and_abstracts.tsv',  # 1
            'disease_pathway.tsv',                           # 2
            'all_diseases.tsv',                              # 3
            'phenotype.hpoa',                                # 4
            'hp.obo',                                        # 5
        ]

    @property
    def processed_file_names(self):
        return [
            'disease_id_feature_index_mapping.tsv',   # 0
            'hpo_id_index_mapping.tsv',               # 1
            'disease_features.pt',                     # 2
            'hpo_features.pt',                         # 3
            'edges.pt',                                # 4
            'data.pt',                                 # 5
        ]

    def download(self):
        for fname in self.raw_file_names:
            dest = osp.join(self.raw_dir, fname)
            if not osp.isfile(dest):
                src = osp.join(self.raw_dir, '..', '..', fname)
                if osp.isfile(src):
                    copyfile(src, dest)

    # ════════════════════════════════════════════════════════════════════════
    # MAIN PROCESS PIPELINE
    # ════════════════════════════════════════════════════════════════════════

    def process(self):
        log.info("=== DiseaseNet: building heterogeneous disease-phenotype graph ===")

        # 1 — Disease index (unchanged logic from original)
        log.info("Step 1/7 — building disease index mapping")
        self.create_disease_index_feature_mapping()
        disease_idx = self.load_disease_index_feature_mapping()
        log.info(f"  {len(disease_idx)} diseases indexed")

        # 2 — Parse hp.obo
        log.info("Step 2/7 — parsing hp.obo")
        terms, alt_map = self._parse_obo(self.obo_path)
        log.info(f"  {len(terms)} HPO terms loaded")

        # 3 — Ancestor table (for IC propagation)
        log.info("Step 3/7 — building ancestor table (iterative DFS, may take ~30s)")
        ancestor_table = self._build_ancestor_table(terms, alt_map)

        # 4 — Load disease→HPO direct annotations
        log.info("Step 4/7 — loading disease-HPO annotations from phenotype.hpoa")
        direct_annotations = self._load_hpoa(disease_idx, alt_map)
        if len(direct_annotations) < 10:
            log.warning("phenotype.hpoa matched very few diseases; "
                        "falling back to disease_hpo.tsv")
            direct_annotations = self._load_disease_hpo_tsv(disease_idx, alt_map)
        log.info(f"  annotations loaded for {len(direct_annotations)} diseases")

        # 5 — Compute IC per HPO term
        log.info("Step 5/7 — computing information content (IC) scores")
        n_dis      = len(disease_idx)
        ic_scores  = self._compute_ic(direct_annotations, ancestor_table, n_dis)
        if ic_scores:
            log.info(
                f"  IC computed for {len(ic_scores)} terms | "
                f"max={max(ic_scores.values()):.2f}  "
                f"mean={sum(ic_scores.values())/len(ic_scores):.2f}"
            )

        # Build HPO node index: one node per term appearing as a direct annotation
        all_direct_hpo = sorted({h for s in direct_annotations.values() for h in s})
        hpo_idx        = {hp_id: i for i, hp_id in enumerate(all_direct_hpo)}
        log.info(f"  {len(hpo_idx)} unique direct HPO terms → HPO nodes")

        # Save HPO index mapping
        with open(self.processed_paths[self.ProcessedFileEnum.hpo_id_index_mapping],
                  'w') as fh:
            fh.write('hpo_id\tindex\n')
            for hp_id, i in hpo_idx.items():
                fh.write(f'{hp_id}\t{i}\n')

        # Collect direct annotation counts per HPO term (used in HPO features)
        direct_hpo_count = defaultdict(int)
        for hpo_set in direct_annotations.values():
            for hp_id in hpo_set:
                direct_hpo_count[hp_id] += 1

        # 6 — Node feature matrices
        log.info("Step 6/7 — building node feature matrices")
        self._build_disease_features(disease_idx)
        self._build_hpo_features(hpo_idx, ic_scores, terms, direct_hpo_count, n_dis)

        # 7 — Heterogeneous edges
        log.info("Step 7/7 — building heterogeneous edges")
        self._build_all_edges(
            disease_idx, hpo_idx, direct_annotations, ic_scores, terms, alt_map
        )

        # Assemble and save HeteroData
        log.info("Assembling HeteroData")
        self._assemble_hetero_data()
        log.info("=== DiseaseNet: done ===")

    # ════════════════════════════════════════════════════════════════════════
    # OBO PARSING
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_obo(obo_path: str):
        """
        Parse hp.obo and return:
          terms   — {hp_id → {'name', 'parents': [hp_id,...], 'is_obsolete',
                               'alt_ids': [str,...]}}
          alt_map — {alt_id → canonical hp_id}
        """
        terms   = {}
        alt_map = {}
        current = None

        with open(obo_path, 'r', encoding='utf-8') as fh:
            for raw in fh:
                line = raw.strip()

                if line == '[Term]':
                    if current and 'id' in current:
                        terms[current['id']] = current
                        for a in current['alt_ids']:
                            alt_map[a] = current['id']
                    current = {
                        'parents': [], 'alt_ids': [],
                        'is_obsolete': False, 'name': '',
                    }

                elif line.startswith('[') and line != '[Term]':
                    # End of term stanza (e.g. [Typedef])
                    if current and 'id' in current:
                        terms[current['id']] = current
                        for a in current['alt_ids']:
                            alt_map[a] = current['id']
                    current = None

                elif current is not None:
                    if line.startswith('id: '):
                        current['id'] = line[4:].strip()
                    elif line.startswith('name: '):
                        current['name'] = line[6:].strip()
                    elif line.startswith('is_a: '):
                        # "is_a: HP:0000001 ! Some parent term"
                        parent = line[6:].split('!')[0].strip()
                        current['parents'].append(parent)
                    elif line.startswith('alt_id: '):
                        current['alt_ids'].append(line[8:].strip())
                    elif line == 'is_obsolete: true':
                        current['is_obsolete'] = True

        # Flush last stanza
        if current and 'id' in current:
            terms[current['id']] = current
            for a in current['alt_ids']:
                alt_map[a] = current['id']

        return terms, alt_map

    @staticmethod
    def _build_ancestor_table(terms: dict, alt_map: dict) -> dict:
        """
        For every HP term compute frozenset of all ancestor IDs (including self).
        Uses iterative DFS so it never hits Python's recursion limit.

        Returns {hp_id → frozenset(ancestor_hp_ids including self)}
        """
        cache = {}

        for start in list(terms.keys()):
            if start in cache:
                continue
            visited = set()
            stack   = [alt_map.get(start, start)]
            while stack:
                node = alt_map.get(stack.pop(), None)
                if node is None or node in visited:
                    continue
                visited.add(node)
                t = terms.get(node)
                if t and not t.get('is_obsolete', False):
                    for p in t['parents']:
                        p = alt_map.get(p, p)
                        if p not in visited:
                            stack.append(p)
            cache[start] = frozenset(visited)

        return cache

    # ════════════════════════════════════════════════════════════════════════
    # ANNOTATION LOADING
    # ════════════════════════════════════════════════════════════════════════

    def _load_hpoa(self, disease_idx: dict, alt_map: dict) -> dict:
        """
        Load direct disease-HPO annotations from phenotype.hpoa.

        Supports both the current format (single 'DatabaseID' column) and the
        legacy format ('disease-db' + 'disease-db-id' columns).

        Filters applied:
          * OMIM diseases only
          * Qualifier must not be 'NOT'
          * Aspect must be 'P' (phenotypic abnormality)

        Returns {disease_key → set of direct HPO term IDs (canonical)}
        """
        if not osp.isfile(self.hpoa_path):
            log.warning(f"phenotype.hpoa not found at {self.hpoa_path}")
            return defaultdict(set)

        known  = set(disease_idx.keys())
        result = defaultdict(set)
        cols   = None

        with open(self.hpoa_path, 'r', encoding='utf-8') as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue

                # Header (starts with '#')
                if line.startswith('#'):
                    cols = {h.strip(): i
                            for i, h in enumerate(line.lstrip('#').split('\t'))}
                    continue
                if cols is None:
                    continue

                parts = line.split('\t')

                try:
                    # ── current format: DatabaseID = 'OMIM:614021' ──────────
                    if 'DatabaseID' in cols:
                        db_id  = parts[cols['DatabaseID']].strip()
                        qual   = (parts[cols['Qualifier']].strip()
                                  if 'Qualifier' in cols and cols['Qualifier'] < len(parts)
                                  else '')
                        hpo_id = (parts[cols['HPO_ID']].strip()
                                  if 'HPO_ID' in cols and cols['HPO_ID'] < len(parts)
                                  else '')
                        aspect = (parts[cols['Aspect']].strip()
                                  if 'Aspect' in cols and cols['Aspect'] < len(parts)
                                  else 'P')

                    # ── legacy format: 'disease-db' + 'disease-db-id' ───────
                    elif 'disease-db' in cols and 'disease-db-id' in cols:
                        db_id  = (f"{parts[cols['disease-db']].strip()}:"
                                  f"{parts[cols['disease-db-id']].strip()}")
                        qi     = cols.get('qualifier', 999)
                        hi     = cols.get('hpo-id',   999)
                        ai     = cols.get('aspect',   999)
                        qual   = parts[qi].strip() if qi < len(parts) else ''
                        hpo_id = parts[hi].strip() if hi < len(parts) else ''
                        aspect = parts[ai].strip() if ai < len(parts) else 'P'
                    else:
                        continue

                except IndexError:
                    continue

                # Apply filters
                if (not db_id.startswith('OMIM:')
                        or qual == 'NOT'
                        or aspect != 'P'
                        or not hpo_id):
                    continue

                # Resolve disease key (try 'OMIM:614021' then '614021')
                key = db_id if db_id in known else db_id.split(':', 1)[-1]
                if key not in known:
                    continue

                result[key].add(alt_map.get(hpo_id, hpo_id))

        return result

    def _load_disease_hpo_tsv(self, disease_idx: dict, alt_map: dict) -> dict:
        """Fallback: load annotations from the existing disease_hpo.tsv."""
        result = defaultdict(set)
        path   = osp.join(self.raw_dir, 'disease_hpo.tsv')
        if not osp.isfile(path):
            return result
        with open(path, 'r') as fh:
            for line in fh:
                parts = [s.strip() for s in line.split('\t')]
                if len(parts) < 2:
                    continue
                dis_id, hpo_id = parts[0], parts[1]
                hpo_id = alt_map.get(hpo_id, hpo_id)
                if dis_id in disease_idx:
                    result[dis_id].add(hpo_id)
        return result

    # ════════════════════════════════════════════════════════════════════════
    # INFORMATION CONTENT
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_ic(direct_annotations: dict,
                    ancestor_table: dict,
                    total_diseases: int) -> dict:
        """
        Propagate disease annotations upward via the true-path rule, then
        compute IC(t) = -log( freq(t) / total_diseases ).

        A term's frequency after propagation counts any disease that is
        directly annotated with it *or any of its descendants*.  Rare,
        specific terms therefore receive high IC values; root-level terms
        that annotate almost everything receive IC ≈ 0.

        Returns {hp_id → float IC value}
        """
        term_count = defaultdict(int)
        for hpo_set in direct_annotations.values():
            propagated = set()
            for hp_id in hpo_set:
                propagated |= ancestor_table.get(hp_id, frozenset({hp_id}))
            for anc in propagated:
                term_count[anc] += 1

        ic = {}
        for hp_id, cnt in term_count.items():
            freq = cnt / max(total_diseases, 1)
            ic[hp_id] = -math.log(freq) if freq > 0 else 0.0
        return ic

    # ════════════════════════════════════════════════════════════════════════
    # FEATURE MATRICES
    # ════════════════════════════════════════════════════════════════════════

    def _build_disease_features(self, disease_idx: dict):
        """
        Disease node features: pathway one-hot + TF-IDF publication features.

        HPO binary features are deliberately excluded: phenotype information
        now flows through the disease→HPO→disease graph edges, where IC
        weighting correctly values rare shared phenotypes.
        """
        dis_sorted = sorted(disease_idx, key=disease_idx.__getitem__)
        x = None

        if 'pathways' in self.feature_source:
            log.info("  disease features: pathway one-hot")
            mlb         = MultiLabelBinarizer()
            pathway_ids = set()
            dis_pathway = defaultdict(set)
            pw_path     = osp.join(self.raw_dir, 'disease_pathway.tsv')
            if osp.isfile(pw_path):
                with open(pw_path) as fh:
                    for line in fh:
                        p = [s.strip() for s in line.split('\t')]
                        if len(p) >= 2:
                            pathway_ids.add(p[1])
                            dis_pathway[p[0]].add(p[1])
            mlb.fit([sorted(pathway_ids)])
            x = torch.tensor(
                mlb.transform([sorted(dis_pathway[d]) for d in dis_sorted]),
                dtype=torch.float,
            )

        if 'disease_publications' in self.feature_source:
            log.info("  disease features: TF-IDF publications")
            dis_texts = defaultdict(str)
            corpus    = []
            pub_path  = osp.join(self.raw_dir,
                                  'disease_publication_titles_and_abstracts.tsv')
            if osp.isfile(pub_path):
                with open(pub_path, 'r', encoding='utf-8') as fh:
                    for line in fh:
                        p = [s.strip() for s in line.split('\t')]
                        if len(p) < 3:
                            continue
                        did, title, abstract = p[0], p[1], p[2]
                        if abstract:
                            corpus.append(abstract)
                            dis_texts[did] += f' {abstract}'
                        if title:
                            corpus.append(title)
                            dis_texts[did] += f' {title}'
                vect = TfidfVectorizer(ngram_range=(1, 2),
                                       max_df=0.01, min_df=0.001)
                vect.fit(corpus)
                tmp = torch.tensor(
                    vect.transform([dis_texts[d] for d in dis_sorted]).toarray(),
                    dtype=torch.float,
                )
                x = torch.cat([x, tmp], dim=1) if x is not None else tmp

        if x is None:
            log.warning("No disease feature source produced features; "
                        "using small zero-initialised placeholder.")
            x = torch.zeros(len(dis_sorted), 8, dtype=torch.float)

        if not self.skip_truncated_svd and x.shape[1] > self.svd_components:
            log.info(f"  TruncatedSVD: {x.shape[1]} → {self.svd_components} dims")
            svd = TruncatedSVD(n_components=self.svd_components,
                               n_iter=self.svd_n_iter,
                               random_state=self.seed)
            x = torch.tensor(svd.fit_transform(x.numpy()), dtype=torch.float)

        torch.save(x, self.processed_paths[self.ProcessedFileEnum.disease_features])
        log.info(f"  disease feature matrix: {tuple(x.shape)}")

    def _build_hpo_features(self, hpo_idx: dict, ic_scores: dict,
                             terms: dict, direct_hpo_count: dict,
                             total_diseases: int):
        """
        HPO node features — shape [N_hpo, 3]:
          col 0: IC_norm            = IC(t) / max_IC
          col 1: depth_norm         = BFS depth from root / max_depth
          col 2: log_freq_norm      = log(1 + direct_annotation_count) /
                                      log(1 + total_diseases)

        Depth is computed via BFS from HP:0000001 using the is-a parent map.
        """
        # Build parent → children map for efficient BFS
        parent_to_children = defaultdict(list)
        for hp_id, term in terms.items():
            if not term.get('is_obsolete', False):
                for p in term['parents']:
                    parent_to_children[p].append(hp_id)

        root      = 'HP:0000001'
        depth_map = {root: 0}
        queue     = collections.deque([root])
        while queue:
            node = queue.popleft()
            for child in parent_to_children.get(node, []):
                if child not in depth_map:
                    depth_map[child] = depth_map[node] + 1
                    queue.append(child)

        max_depth = max(depth_map.values()) if depth_map else 1
        max_ic    = max(ic_scores.values())  if ic_scores else 1.0
        log_denom = math.log(1.0 + total_diseases)

        rows = []
        for hp_id in sorted(hpo_idx, key=hpo_idx.__getitem__):
            ic    = ic_scores.get(hp_id, 0.0)
            depth = depth_map.get(hp_id, 0)
            cnt   = direct_hpo_count.get(hp_id, 0)
            rows.append([
                ic    / max_ic    if max_ic    > 0 else 0.0,
                depth / max_depth if max_depth > 0 else 0.0,
                math.log(1.0 + cnt) / log_denom if log_denom > 0 else 0.0,
            ])

        x_hpo = torch.tensor(rows, dtype=torch.float)
        torch.save(x_hpo, self.processed_paths[self.ProcessedFileEnum.hpo_features])
        log.info(f"  HPO feature matrix: {tuple(x_hpo.shape)}")

    # ════════════════════════════════════════════════════════════════════════
    # EDGE CONSTRUCTION
    # ════════════════════════════════════════════════════════════════════════

    def _build_all_edges(self, disease_idx: dict, hpo_idx: dict,
                         direct_annotations: dict, ic_scores: dict,
                         terms: dict, alt_map: dict):
        """
        Build all four edge sets and save as:
            {
              'dis_hpo' : (edge_index [2,E], edge_attr [E,1]),
              'hpo_dis' : (edge_index [2,E], edge_attr [E,1]),
              'hpo_hpo' : (edge_index [2,E], edge_attr [E,1]),
              'dis_dis' : (edge_index [2,E], edge_attr [E,1]),
            }
        """
        edges = {}

        # ── 1. Disease → HPO annotation edges ────────────────────────────
        log.info("  building disease ↔ HPO annotation edges (IC-weighted)")
        src_dh, tgt_dh, w_dh = [], [], []

        for dis_id, hpo_set in direct_annotations.items():
            d = disease_idx.get(dis_id)
            if d is None:
                continue
            for hp_id in hpo_set:
                h = hpo_idx.get(hp_id)
                if h is None:
                    continue
                src_dh.append(d)
                tgt_dh.append(h)
                w_dh.append(ic_scores.get(hp_id, 0.0))

        if src_dh:
            ei_dh = torch.tensor([src_dh, tgt_dh], dtype=torch.long)
            ea_dh = torch.tensor(w_dh, dtype=torch.float).unsqueeze(1)
            # Forward: disease → hpo
            edges['dis_hpo'] = (ei_dh, ea_dh)
            # Reverse: hpo → disease  (same IC weights)
            edges['hpo_dis'] = (
                torch.tensor([tgt_dh, src_dh], dtype=torch.long), ea_dh
            )
            log.info(f"    {len(src_dh)} disease→HPO annotation edges built")
        else:
            log.warning("    no disease→HPO edges built — check HPOA loading")
            _e = (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 1))
            edges['dis_hpo'] = _e
            edges['hpo_dis'] = _e

        # ── 2. HPO → HPO is-a ontology edges ─────────────────────────────
        log.info("  building HPO is-a ontology edges")
        src_hh, tgt_hh = [], []

        for hp_id, term in terms.items():
            if term.get('is_obsolete', False):
                continue
            child_i = hpo_idx.get(hp_id)
            if child_i is None:
                continue                        # not in annotation set → skip
            for parent_id in term['parents']:
                parent_id  = alt_map.get(parent_id, parent_id)
                parent_i   = hpo_idx.get(parent_id)
                if parent_i is None:
                    continue                    # parent not in annotation set
                src_hh.append(child_i)
                tgt_hh.append(parent_i)

        if src_hh:
            edges['hpo_hpo'] = (
                torch.tensor([src_hh, tgt_hh], dtype=torch.long),
                torch.ones(len(src_hh), 1, dtype=torch.float),
            )
            log.info(f"    {len(src_hh)} HPO→HPO is-a ontology edges built")
        else:
            edges['hpo_hpo'] = (torch.zeros(2, 0, dtype=torch.long),
                                 torch.zeros(0, 1))

        # ── 3. Disease → Disease IC-weighted Resnik similarity edges ──────
        log.info("  building IC-weighted disease ↔ disease similarity edges")

        # Index HPO term → set of disease indices for fast pair enumeration
        hpo_to_dis_set = defaultdict(set)
        for dis_id, hpo_set in direct_annotations.items():
            d = disease_idx.get(dis_id)
            if d is None:
                continue
            for hp_id in hpo_set:
                hpo_to_dis_set[hp_id].add(d)

        # Accumulate sum-IC (Resnik-style) over shared informative terms
        pair_sim: dict = defaultdict(float)
        for hp_id, dis_set in hpo_to_dis_set.items():
            ic_val = ic_scores.get(hp_id, 0.0)
            if ic_val < self.min_ic_for_disease_edge:
                continue                        # filter out root-level / uninformative
            dis_list = sorted(dis_set)
            for i, d1 in enumerate(dis_list):
                for d2 in dis_list[i + 1:]:
                    pair_sim[(d1, d2)] += ic_val

        # Optional: supplement with GDA gene-sharing edges
        if self.gda_edges and osp.isfile(self.train_positives_path):
            log.info("  adding GDA-derived disease↔disease gene-sharing edges")
            gda = pd.read_csv(self.train_positives_path, sep='\t',
                              names=['EntrezGene ID', 'OMIM ID', 'Score'])
            dis2genes: dict = defaultdict(set)
            for _, row in gda.iterrows():
                dis2genes[row['OMIM ID']].add(row['EntrezGene ID'])
            dis_keys = sorted(dis2genes.keys())
            for i, d1_id in enumerate(dis_keys):
                d1 = disease_idx.get(d1_id)
                if d1 is None:
                    continue
                for d2_id in dis_keys[i + 1:]:
                    d2 = disease_idx.get(d2_id)
                    if d2 is None:
                        continue
                    shared = len(dis2genes[d1_id] & dis2genes[d2_id])
                    if shared > 0:
                        key = (min(d1, d2), max(d1, d2))
                        pair_sim[key] += float(shared)

        # Build symmetric undirected edge list
        src_dd, tgt_dd, w_dd = [], [], []
        for (d1, d2), sim in pair_sim.items():
            src_dd += [d1, d2]
            tgt_dd += [d2, d1]
            w_dd   += [sim, sim]

        if src_dd:
            order  = sorted(range(len(src_dd)),
                            key=lambda i: (src_dd[i], tgt_dd[i]))
            src_dd = [src_dd[k] for k in order]
            tgt_dd = [tgt_dd[k] for k in order]
            w_dd   = [w_dd[k]   for k in order]
            edges['dis_dis'] = (
                torch.tensor([src_dd, tgt_dd], dtype=torch.long),
                torch.tensor(w_dd, dtype=torch.float).unsqueeze(1),
            )
            log.info(f"    {len(src_dd) // 2} unique disease↔disease edges built")
        else:
            edges['dis_dis'] = (torch.zeros(2, 0, dtype=torch.long),
                                 torch.zeros(0, 1))

        torch.save(edges, self.processed_paths[self.ProcessedFileEnum.edges])

    # ════════════════════════════════════════════════════════════════════════
    # HETERODATA ASSEMBLY
    # ════════════════════════════════════════════════════════════════════════

    def _assemble_hetero_data(self):
        x_dis  = torch.load(self.processed_paths[self.ProcessedFileEnum.disease_features])
        x_hpo  = torch.load(self.processed_paths[self.ProcessedFileEnum.hpo_features])
        edges  = torch.load(self.processed_paths[self.ProcessedFileEnum.edges])

        data = HeteroData()

        # Node features
        data['disease'].x         = x_dis
        data['disease'].num_nodes = x_dis.size(0)
        data['hpo'].x             = x_hpo
        data['hpo'].num_nodes     = x_hpo.size(0)

        # Edge: disease → HPO (IC-weighted annotation)
        ei, ea = edges['dis_hpo']
        data['disease', 'annotated_with', 'hpo'].edge_index = ei
        data['disease', 'annotated_with', 'hpo'].edge_attr  = ea

        # Edge: HPO → disease (reverse annotation)
        ei, ea = edges['hpo_dis']
        data['hpo', 'rev_annotated', 'disease'].edge_index  = ei
        data['hpo', 'rev_annotated', 'disease'].edge_attr   = ea

        # Edge: HPO → HPO (is-a ontology hierarchy)
        ei, ea = edges['hpo_hpo']
        data['hpo', 'is_a', 'hpo'].edge_index = ei
        data['hpo', 'is_a', 'hpo'].edge_attr  = ea

        # Edge: disease → disease (IC-weighted Resnik similarity)
        ei, ea = edges['dis_dis']
        data['disease', 'pheno_sim', 'disease'].edge_index = ei
        data['disease', 'pheno_sim', 'disease'].edge_attr  = ea

        # Collate into InMemoryDataset format and save
        data_list          = [data]
        data, slices       = self.collate(data_list)
        torch.save((data, slices),
                   self.processed_paths[self.ProcessedFileEnum.data])
        log.info(
            f"HeteroData saved — disease nodes: {x_dis.shape}  "
            f"HPO nodes: {x_hpo.shape}"
        )

    # ════════════════════════════════════════════════════════════════════════
    # DISEASE INDEX MAPPING  (logic unchanged from original)
    # ════════════════════════════════════════════════════════════════════════

    def create_disease_index_feature_mapping(self):
        disease_index_mapping = collections.OrderedDict()

        # Seed from all_diseases.tsv
        all_dis_path = osp.join(self.raw_dir, 'all_diseases.tsv')
        if osp.isfile(all_dis_path):
            with open(all_dis_path, 'r') as fh:
                for line in fh:
                    parts      = [s.strip() for s in line.split('\t')]
                    identifier = parts[0]
                    if (identifier.startswith('OMIM')
                            and identifier not in disease_index_mapping):
                        disease_index_mapping[identifier] = len(disease_index_mapping)

        # Training / negative files
        for dpath in [self.train_positives_path, self.train_negatives_path]:
            if dpath and dpath != 'random' and osp.isfile(dpath):
                df = pd.read_csv(dpath, sep='\t', header=None)
                for disease in df[1].drop_duplicates().tolist():
                    if disease not in disease_index_mapping:
                        disease_index_mapping[disease] = len(disease_index_mapping)

        # UMLS → OMIM mapping
        if osp.isfile(self.umls_omim_mapping_path):
            mapping_df = pd.read_csv(self.umls_omim_mapping_path,
                                     sep='\t', header=None, dtype=str)
            for _, row in mapping_df.iterrows():
                try:
                    omim_ids = ast.literal_eval(row[1])
                except Exception:
                    continue
                for omim in omim_ids:
                    omim = str(omim).strip()
                    if omim and omim not in disease_index_mapping:
                        disease_index_mapping[omim] = len(disease_index_mapping)

        out_path = self.processed_paths[
            self.ProcessedFileEnum.disease_id_feature_index_mapping
        ]
        with open(out_path, 'w') as fh:
            fh.write('disease_id\tindex\n')
            for did, idx in disease_index_mapping.items():
                fh.write(f'{did}\t{idx}\n')

    def load_disease_index_feature_mapping(self) -> dict:
        mapping = collections.OrderedDict()
        path    = self.processed_paths[
            self.ProcessedFileEnum.disease_id_feature_index_mapping
        ]
        with open(path, 'r') as fh:
            next(fh)
            for line in fh:
                did, idx = [s.strip() for s in line.split('\t')]
                mapping[did] = int(idx)
        return mapping

    def load_hpo_index_mapping(self) -> dict:
        """Convenience accessor — returns {hpo_id → int index}."""
        mapping = collections.OrderedDict()
        path    = self.processed_paths[self.ProcessedFileEnum.hpo_id_index_mapping]
        with open(path, 'r') as fh:
            next(fh)
            for line in fh:
                hpo_id, idx = [s.strip() for s in line.split('\t')]
                mapping[hpo_id] = int(idx)
        return mapping