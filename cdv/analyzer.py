"""CDV Analyzer — NMF/Lasso pipeline for Controlled Domain Vocabulary discovery.

Phase 4 Steps 2-7:
  2. Strip Layer 1+2 vocabulary from corpus
  3. TF-IDF matrix over remaining words
  4. NMF analysis → latent intent dimensions
  5. Lasso regression → minimum predictive word set
  6. Mutual information scoring → rank by potency
  7. Final CDV delta → ~200 highest-potency words
"""

import json
import keyword
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


class CDVAnalyzer:
    """Discover the Controlled Domain Vocabulary for a specialist domain."""

    def __init__(self, corpus_path: str, output_dir: str = "cdv"):
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Layer 1: Python language keywords
        self.layer1 = set(keyword.kwlist) | {"True", "False", "None", "self", "cls"}

        # Layer 2: Python builtins ecosystem
        self.layer2 = {
            "list", "dict", "str", "int", "float", "bool", "tuple", "set",
            "sorted", "filter", "map", "range", "len", "type", "print",
            "input", "open", "read", "write", "close", "file", "path",
            "import", "from", "as", "with", "def", "class", "return", "yield",
            "lambda", "try", "except", "finally", "raise", "assert",
            "if", "elif", "else", "for", "while", "break", "continue", "pass",
            "and", "or", "not", "in", "is", "None", "True", "False",
            "async", "await", "nonlocal", "global", "del",
            "object", "super", "init", "main", "name", "value", "key",
            "item", "data", "result", "function", "method", "argument",
            "parameter", "variable", "python", "code", "number", "string",
            "integer", "boolean", "array", "element", "index", "count",
            "sum", "min", "max", "abs", "all", "any", "enumerate", "zip",
            "reversed", "next", "iter", "format", "join", "split", "replace",
            "strip", "lower", "upper", "startswith", "endswith", "find",
            "append", "extend", "insert", "remove", "pop", "clear", "copy",
            "get", "keys", "values", "items", "update", "setdefault",
            "hasattr", "getattr", "setattr", "isinstance", "issubclass",
            "callable", "dir", "vars", "help", "id", "hash", "repr", "str",
        }
        self.known = self.layer1 | self.layer2

        # State
        self.corpus: list[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.nmf: Optional[NMF] = None
        self.nmf_topics: list[list[tuple[str, float]]] = []
        self.lasso: Optional[LassoCV] = None
        self.lasso_words: list[tuple[str, float]] = []
        self.mi_scores: list[tuple[str, float]] = []
        self.final_cdv: list[dict] = []

    # ------------------------------------------------------------------
    # Step 1: Load corpus
    # ------------------------------------------------------------------

    def load_corpus(self) -> int:
        """Load the task description corpus from a JSONL file."""
        self.corpus = []
        with open(self.corpus_path) as f:
            for line in f:
                try:
                    task = json.loads(line)
                    text = task.get("text", "").strip()
                    if len(text) > 10:
                        self.corpus.append(text)
                except json.JSONDecodeError:
                    continue
        print(f"Loaded corpus: {len(self.corpus)} task descriptions")
        return len(self.corpus)

    # ------------------------------------------------------------------
    # Step 2: Strip known vocabulary
    # ------------------------------------------------------------------

    def strip_known_vocabulary(self, text: str) -> str:
        """Remove Layer 1 + Layer 2 words from a text, keeping only candidate intent words."""
        # Tokenize: keep only alphabetic words, lowercase
        words = re.findall(r"[a-z_]+", text.lower())
        # Remove known vocabulary
        unknown = [w for w in words if w not in self.known and len(w) > 1]
        return " ".join(unknown)

    def build_stripped_corpus(self) -> list[str]:
        """Build a version of the corpus with known vocabulary removed."""
        return [self.strip_known_vocabulary(t) for t in self.corpus]

    # ------------------------------------------------------------------
    # Step 3: TF-IDF matrix
    # ------------------------------------------------------------------

    def build_tfidf(self, stripped_corpus: list[str], max_features: int = 5000):
        """Build TF-IDF matrix from the stripped corpus."""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,          # appear in at least 2 documents
            max_df=0.5,        # don't appear in more than 50% of documents
            stop_words=None,   # we already stripped known vocab
            ngram_range=(1, 2),  # unigrams and bigrams
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(stripped_corpus)
        vocab = self.vectorizer.get_feature_names_out()
        print(f"TF-IDF matrix: {self.tfidf_matrix.shape[0]} docs × {self.tfidf_matrix.shape[1]} features")
        return self.tfidf_matrix

    # ------------------------------------------------------------------
    # Step 4: NMF analysis
    # ------------------------------------------------------------------

    def run_nmf(self, n_components: int = 20, top_n: int = 15):
        """Run NMF to discover latent intent dimensions."""
        self.nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
        W = self.nmf.fit_transform(self.tfidf_matrix)  # doc-topic matrix
        H = self.nmf.components_                        # topic-feature matrix

        vocab = self.vectorizer.get_feature_names_out()
        self.nmf_topics = []

        for topic_idx in range(n_components):
            top_indices = H[topic_idx].argsort()[-top_n:][::-1]
            top_words = [(vocab[i], float(H[topic_idx][i])) for i in top_indices]
            self.nmf_topics.append(top_words)

        print(f"NMF: {n_components} topics, {top_n} words/topic")
        return self.nmf_topics

    # ------------------------------------------------------------------
    # Step 5: Lasso regression
    # ------------------------------------------------------------------

    def run_lasso(self, target_feature: Optional[str] = None):
        """Run LassoCV to find minimum word set that predicts code output.

        Since we don't have a direct code-output target for each corpus entry,
        we use the TF-IDF scores as a proxy: Lasso identifies the sparsest
        set of words that can reconstruct the TF-IDF matrix.
        """
        self.lasso = LassoCV(
            cv=5,
            random_state=42,
            max_iter=2000,
            n_alphas=50,
        )

        # Use the mean TF-IDF vector as a proxy target
        y = np.asarray(self.tfidf_matrix.mean(axis=1)).ravel()
        X = self.tfidf_matrix.toarray()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.lasso.fit(X_scaled, y)

        vocab = self.vectorizer.get_feature_names_out()
        coef = self.lasso.coef_

        # Non-zero coefficients = important words
        nonzero = np.abs(coef) > 1e-6
        self.lasso_words = [
            (vocab[i], float(coef[i]))
            for i in range(len(vocab))
            if nonzero[i]
        ]
        self.lasso_words.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"Lasso: {np.sum(nonzero)} non-zero coefficients (alpha={self.lasso.alpha_:.4f})")
        return self.lasso_words

    # ------------------------------------------------------------------
    # Step 6: Mutual information scoring
    # ------------------------------------------------------------------

    def run_mutual_info(self):
        """Compute mutual information between each word and TF-IDF-based signal."""
        X = self.tfidf_matrix.toarray()
        y = np.asarray(self.tfidf_matrix.mean(axis=1)).ravel()

        # Discretize y into bins for MI calculation
        y_binned = np.digitize(y, bins=np.percentile(y, [25, 50, 75]))

        mi = mutual_info_regression(X, y_binned, random_state=42)

        vocab = self.vectorizer.get_feature_names_out()
        self.mi_scores = [
            (vocab[i], float(mi[i]))
            for i in range(len(vocab))
            if mi[i] > 0
        ]
        self.mi_scores.sort(key=lambda x: x[1], reverse=True)

        print(f"Mutual information: {len(self.mi_scores)} words with MI > 0")
        return self.mi_scores

    # ------------------------------------------------------------------
    # Step 7: Final CDV delta
    # ------------------------------------------------------------------

    def build_final_cdv(self, target_words: int = 200) -> list[dict]:
        """Combine NMF, Lasso, and MI scores into final CDV with potency rankings.

        Scoring strategy:
        - NMF: average topic weight across all topics
        - Lasso: normalized absolute coefficient
        - MI: normalized mutual information score
        - Potency: weighted average of all three
        """
        vocab = self.vectorizer.get_feature_names_out()

        # Build NMF scores
        nmf_scores = {}
        for topic in self.nmf_topics:
            for word, score in topic:
                if word not in nmf_scores:
                    nmf_scores[word] = []
                nmf_scores[word].append(score)
        nmf_avg = {w: np.mean(s) for w, s in nmf_scores.items()}

        # Build Lasso scores
        lasso_scores = {w: abs(c) for w, c in self.lasso_words}

        # Build MI scores
        mi_scores = dict(self.mi_scores)

        # Normalize each
        def normalize(d):
            if not d:
                return {}
            max_v = max(d.values())
            return {k: v / max_v for k, v in d.items()} if max_v > 0 else d

        nmf_norm = normalize(nmf_avg)
        lasso_norm = normalize(lasso_scores)
        mi_norm = normalize(mi_scores)

        # Compute composite potency
        all_words = set(nmf_norm) | set(lasso_norm) | set(mi_norm)

        cdv_entries = []
        for word in all_words:
            n = nmf_norm.get(word, 0)
            l = lasso_norm.get(word, 0)
            m = mi_norm.get(word, 0)

            # Weight: NMF 30%, Lasso 40%, MI 30%
            potency = 0.3 * n + 0.4 * l + 0.3 * m

            cdv_entries.append({
                "word": word,
                "potency": round(potency, 4),
                "nmf_score": round(n, 4),
                "lasso_score": round(l, 4),
                "mi_score": round(m, 4),
                "appears_in_topics": len(nmf_scores.get(word, [])),
            })

        # Sort by potency descending
        cdv_entries.sort(key=lambda x: x["potency"], reverse=True)

        # Take top N
        self.final_cdv = cdv_entries[:target_words]

        print(f"Final CDV: {len(self.final_cdv)} words (target: {target_words})")
        return self.final_cdv

    # ------------------------------------------------------------------
    # Step 10: Document
    # ------------------------------------------------------------------

    def save_cdv(self, filename: str = "python_cdv.json"):
        """Save the final CDV to a JSON file."""
        output_path = self.output_dir / filename

        doc = {
            "specialist": "python",
            "total_words": len(self.final_cdv),
            "layers": {
                "layer1_keywords": len(self.layer1),
                "layer2_ecosystem": len(self.layer2),
                "layer3_intent_delta": len(self.final_cdv),
                "total_vocabulary": len(self.layer1) + len(self.layer2) + len(self.final_cdv),
            },
            "analysis_metadata": {
                "corpus_size": len(self.corpus),
                "tfidf_features": self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
                "nmf_topics": len(self.nmf_topics),
                "lasso_alpha": float(self.lasso.alpha_) if self.lasso else 0,
                "lasso_nonzero": len(self.lasso_words),
                "mi_significant_words": len(self.mi_scores),
            },
            "words": self.final_cdv,
        }

        with open(output_path, "w") as f:
            json.dump(doc, f, indent=2)

        print(f"CDV saved to {output_path}")
        return str(output_path)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self, target_words: int = 200):
        """Run the complete CDV discovery pipeline."""
        print("=" * 60)
        print("CDV Discovery Pipeline")
        print("=" * 60)

        # Step 2
        print("\n[Step 2] Stripping known vocabulary...")
        stripped = self.build_stripped_corpus()
        sample = stripped[0] if stripped else ""
        print(f"  Sample stripped: '{sample[:100]}...'")

        # Step 3
        print("\n[Step 3] Building TF-IDF matrix...")
        self.build_tfidf(stripped)

        # Step 4
        print("\n[Step 4] Running NMF analysis...")
        self.run_nmf(n_components=20, top_n=15)

        # Step 5
        print("\n[Step 5] Running Lasso regression...")
        self.run_lasso()

        # Step 6
        print("\n[Step 6] Computing mutual information...")
        self.run_mutual_info()

        # Step 7
        print(f"\n[Step 7] Building final CDV (target: {target_words} words)...")
        cdv = self.build_final_cdv(target_words=target_words)

        # Show top words
        print(f"\n  Top 20 words by potency:")
        for i, entry in enumerate(cdv[:20]):
            print(f"    {i+1:2d}. {entry['word']:20s}  potency={entry['potency']:.3f}  "
                  f"nmf={entry['nmf_score']:.3f}  lasso={entry['lasso_score']:.3f}  mi={entry['mi_score']:.3f}")

        # Step 10
        print("\n[Step 10] Saving CDV...")
        path = self.save_cdv()

        print(f"\nDone. CDV at: {path}")
        return cdv
