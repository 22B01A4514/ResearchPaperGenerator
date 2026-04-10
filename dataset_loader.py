import json
import os
from tqdm import tqdm
import pickle


class ArxivDatasetLoader:
    """
    Loads and processes the full arXiv dataset from Kaggle
    Dataset: Cornell-University/arxiv (1.7M+ papers)
    File required: arxiv-metadata-oai-snapshot.json
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.dataset_file = os.path.join(data_dir, "arxiv-metadata-oai-snapshot.json")
        self.cache_file = os.path.join(data_dir, "papers_cache.pkl")

        if not os.path.exists(self.dataset_file):
            raise FileNotFoundError(
                f"Dataset file not found at {self.dataset_file}. "
                "Please download and place it inside the data directory."
            )

        os.makedirs(data_dir, exist_ok=True)

    def load_and_process(self, max_papers=None, force_reload=False):
        """
        Load and process arXiv dataset.
        Returns: List of processed paper dictionaries
        """

        # Load from cache if exists
        if os.path.exists(self.cache_file) and not force_reload:
            print(f"Loading papers from cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                papers = pickle.load(f)
            print(f" Loaded {len(papers)} papers from cache")
            return papers

        print(f"Loading dataset from: {self.dataset_file}")
        file_size = os.path.getsize(self.dataset_file) / (1024**3)
        print(f"Dataset size: {file_size:.2f} GB")

        papers = self._load_from_file(max_papers)

        # Cache processed papers
        print("Caching processed papers...")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(papers, f)

        print(f" Cached at {self.cache_file}")
        return papers

    def _load_from_file(self, max_papers=None):
        """
        Load papers line-by-line from JSON file
        """
        papers = []

        print("Reading arXiv dataset...")
        if max_papers:
            print(f"Loading first {max_papers:,} papers...")
        else:
            print("Loading ALL papers (this may take time)...")

        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Processing papers", unit=" papers")):
                if max_papers and i >= max_papers:
                    break

                try:
                    paper = json.loads(line)
                    processed = self._process_paper(paper)
                    if processed:
                        papers.append(processed)
                except (json.JSONDecodeError, KeyError):
                    continue

        print(f"\n Processed {len(papers):,} papers")
        return papers

    def _process_paper(self, paper):
        """
        Convert raw paper into clean structured format
        """
        try:
            return {
                "id": paper.get("id", ""),
                "title": paper.get("title", "").replace('\n', ' ').strip(),
                "authors": paper.get("authors", ""),
                "year": self._extract_year(paper.get("update_date", "")),
                "abstract": paper.get("abstract", "").replace('\n', ' ').strip(),
                "categories": paper.get("categories", ""),
                "domain": self._extract_domain(paper.get("categories", "")),
                "keywords": self._extract_keywords(paper.get("categories", "")),
                "doi": paper.get("doi", "")
            }
        except Exception:
            return None

    def _extract_year(self, date_str):
        """
        Extract year from date string
        """
        if date_str:
            try:
                return int(date_str.split('-')[0])
            except:
                return None
        return None

    def _extract_keywords(self, categories):
        """
        Convert arXiv categories into readable keywords
        """
        if not categories:
            return []

        category_map = {
            'cs.LG': ['machine learning', 'deep learning', 'neural networks'],
            'cs.CL': ['natural language processing', 'NLP', 'language models'],
            'cs.CV': ['computer vision', 'image recognition'],
            'cs.AI': ['artificial intelligence'],
            'stat.ML': ['statistical learning'],
            'cs.IR': ['information retrieval', 'recommendation systems'],
            'cs.RO': ['robotics'],
            'cs.CR': ['cryptography', 'security'],
        }

        keywords = []
        for cat in categories.split():
            if cat in category_map:
                keywords.extend(category_map[cat])

        return list(set(keywords))

    def _extract_domain(self, categories):
        """
        Extract primary domain based on first category
        """
        if not categories:
            return "Other"

        first_cat = categories.split()[0]

        domain_map = {
            'cs.LG': 'Machine Learning',
            'cs.CL': 'Natural Language Processing',
            'cs.CV': 'Computer Vision',
            'cs.AI': 'Artificial Intelligence',
            'stat.ML': 'Statistics & ML',
            'cs.IR': 'Information Retrieval',
            'cs.RO': 'Robotics',
            'cs.CR': 'Security',
        }

        return domain_map.get(first_cat, "Other")

    def get_statistics(self, papers):
        """
        Get dataset statistics
        """
        if not papers:
            return {}

        stats = {
            "total_papers": len(papers),
            "domains": {},
            "year_range": {
                "min": min(p["year"] for p in papers if p["year"]),
                "max": max(p["year"] for p in papers if p["year"]),
            },
            "avg_abstract_length": sum(len(p["abstract"]) for p in papers) / len(papers),
        }

        for paper in papers:
            domain = paper["domain"]
            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1

        return stats
