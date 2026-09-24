from embeddings import embedding_service
from vector_store import SimpleVectorStore


class SemanticMemory:
    """
    User-level semantic memory.

    Each user gets a separate vector store containing
    previous interactions.
    """

    def __init__(self):
        self.users = {}


    def _get_store(
        self,
        user_id: str,
    ):
        if user_id not in self.users:
            self.users[ user_id ] = SimpleVectorStore(
                embedding_service
            )
     # ← returns a SimpleVectorStore instance
        return self.users[
            user_id
        ]


    def add_memory(
        self,
        user_id: str,
        text: str,
    ):
        #2 Fucntions are called to add memory to the vector store for a specific user. 
        #Called as Method Chaining: The add_memory method is called on the SemanticMemory instance, which internally calls the _get_store method to retrieve the appropriate vector store for the user. Then, it calls the add method of the SimpleVectorStore instance to add the memory text. This chaining of method calls allows for a clean and organized way to manage semantic memories for different users.
        self._get_store(user_id).add(text)


    def search(
        self,
        user_id: str,
        query: str,
        k: int = 3,
    ):
        return self._get_store(
            user_id
        ).search(
            query,
            k
        )


semantic_memory = SemanticMemory()
