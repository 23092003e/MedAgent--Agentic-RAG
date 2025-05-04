# prepare_db.py
if __name__ == "__main__":
    import os
    from ..config import DATA_PATH, DB_FAISS_PATH
    from backend.rag.vector_store import create_vector_store

    # Đảm bảo thư mục lưu FAISS index tồn tại
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

    # Tạo vector store
    create_vector_store(DATA_PATH, DB_FAISS_PATH)
    print(f"Vector store created at {DB_FAISS_PATH}")
