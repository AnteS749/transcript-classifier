import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Classifier:
    def __init__(self, txt_path, json_path):
        with open(txt_path, "r") as f:
            self.all_paths = [line.strip() for line in f if line.strip()]
        with open(json_path, "r") as f:
            self.taxonomy_tree = json.load(f)

    def get_valid_paths(self, locked_l0):
        return [
            path for path in self.all_paths
            if path.startswith(f"{locked_l0} >")
            and path.split(" > ")[0] == locked_l0
        ]

    def classify(self, transcript, locked_l0):
        valid_paths = self.get_valid_paths(locked_l0)
        if not valid_paths:
            return {"error": f"Invalid or unsupported locked L0: '{locked_l0}'."}

        corpus = valid_paths + [transcript]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        results = sorted(
            zip(valid_paths, similarities),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        response = []
        for path, score in results:
            score_percent = round(score * 100)
            response.append({"path": path, "score": score_percent})

        strong_scores = [r for r in response if r["score"] >= 50]
        if strong_scores:
            return {"results": response}
        elif response:
            return {"results": [response[0]]}
        else:
            return {"results": []}