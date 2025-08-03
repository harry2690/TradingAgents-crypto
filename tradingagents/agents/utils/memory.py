import chromadb
from chromadb.config import Settings


class FinancialSituationMemory:
    def __init__(self, name, config):
        provider = config["embedding_provider"]
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                base_url=config["embedding_backend_url"],
                api_key=config["embedding_api_key"],
            )
            self.embedding = "text-embedding-3-small"
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(
                api_key=config["embedding_api_key"],
                base_url=config["embedding_backend_url"],
            )
            self.embedding = "claude-embed-1"  # 依實際可用 model 調整
        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=config["embedding_api_key"])
            self.client = genai  # 依該 SDK 的使用方式取得 embeddings
            self.embedding = "textembedding-gecko"  # 範例 model
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        self.provider = provider
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        
        # Make collection name unique per session to avoid conflicts
        session_id = config.get('session_id', 'default')
        unique_name = f"{name}_{session_id}"
        
        # Check if collection already exists, if so delete it and create new one
        try:
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            if unique_name in existing_collections:
                self.chroma_client.delete_collection(name=unique_name)
        except Exception as e:
            # If there's any issue checking/deleting, just continue
            pass
        
        # Create the collection (now guaranteed to be fresh and unique)
        self.situation_collection = self.chroma_client.create_collection(name=unique_name)

    def get_embedding(self, text):
        """Get embedding for a text using the configured provider"""

        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.embedding, input=text
            )
            return response.data[0].embedding
        elif self.provider == "anthropic":
            response = self.client.embeddings.create(
                model=self.embedding, input=text
            )
            return response.data[0].embedding
        elif self.provider == "gemini":
            response = self.client.embed_content(
                model=self.embedding, content=text
            )
            return response["embedding"]
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
