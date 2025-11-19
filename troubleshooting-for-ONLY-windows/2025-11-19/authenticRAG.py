import os
import json
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from openai import OpenAI

# ติดตั้ง library ที่จำเป็น
# pip install opensearchpy sentence-transformers llama-index llama-index-embeddings-huggingface openai

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
# from langchain.schema import Document as LangchainDocument
from langchain_community.document_loaders import TextLoader

from langchain_core.documents import Document as LangchainDocument

class AnthropicStyleContextualRAG:
    def __init__(self, opensearch_host="localhost", opensearch_port=9200):
        # รับ API key จาก environment variable
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")
        
        # สร้าง OpenAI client ที่ใช้เรียก Qwen API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        
        # ตั้งค่า embedding model
        self.embed_model = SentenceTransformer('BAAI/bge-m3')
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        
        # สร้าง wrapper สำหรับ embedding
        class EmbeddingsWrapper:
            def __init__(self, embed_model):
                self.embed_model = embed_model

            def embed_query(self, text):
                return self.embed_model.encode(text).tolist()

            def embed_documents(self, documents):
                return [self.embed_model.encode(doc).tolist() for doc in documents]

        self.encoder = EmbeddingsWrapper(self.embed_model)

        # ตั้งค่าการเชื่อมต่อกับ OpenSearch
        self.opensearch_client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            use_ssl=False
        )

        # ชื่อ indices ใน OpenSearch (แยกระหว่าง vector และ BM25 เหมือนตัวอย่าง)
        self.vector_index_name = "anthropic-vector-index"
        self.bm25_index_name = "anthropic-bm25-index"

        # ค่า k สำหรับ RRF
        self.rrf_k = 60

        # ตั้งค่า MarkdownNodeParser
        self.md_parser = MarkdownNodeParser(chunk_size=256)
        
        # สร้าง indices ถ้ายังไม่มี
        self._create_or_update_indices()
    
    def _create_or_update_indices(self):
        """สร้างหรืออัปเดต indices ที่ใช้ในการเก็บ embeddings และ BM25"""
        # 1. สร้าง vector index
        if not self.opensearch_client.indices.exists(index=self.vector_index_name):
            vector_settings = {
                "settings": {
                    "index.knn": True,
                    "index.knn.space_type": "cosinesimil"
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.embedding_dim
                        },
                        "doc_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "chunk_id": {"type": "keyword"}
                    }
                }
            }
            self.opensearch_client.indices.create(index=self.vector_index_name, body=vector_settings)
            print(f"สร้าง vector index {self.vector_index_name} สำเร็จ")
        
        # 2. สร้าง BM25 index
        if not self.opensearch_client.indices.exists(index=self.bm25_index_name):
            bm25_settings = {
                "settings": {
                    "analysis": {"analyzer": {"default": {"type": "standard"}}},
                    "similarity": {"default": {"type": "BM25"}},
                    "index.queries.cache.enabled": False
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "standard"},
                        "contextualized_content": {"type": "text", "analyzer": "standard"},
                        "doc_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "original_index": {"type": "integer", "index": False}
                    }
                }
            }
            self.opensearch_client.indices.create(index=self.bm25_index_name, body=bm25_settings)
            print(f"สร้าง BM25 index {self.bm25_index_name} สำเร็จ")

    def call_qwen_api(self, prompt, max_tokens=500, temperature=0.2):
        """เรียกใช้ Qwen API เพื่อสร้างข้อความ"""
        try:
            completion = self.client.chat.completions.create(
                model="qwen2.5-32b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Qwen API: {e}")
            return f"Qwen API error: {str(e)}"

    def load_documents(self, md_paths):
        """โหลดเอกสาร Markdown และแบ่งเป็น chunks"""
        print(f"Loading documents from {len(md_paths)} files...")
        
        # อ่านไฟล์ด้วย TextLoader
        # pages = []
        # for path in md_paths:
        #     loader = TextLoader(path)
        #     pages.extend(loader.load())

        pages = []
        for path in md_paths:
            # เพิ่ม encoding="utf-8" เข้าไปใน TextLoader
            loader = TextLoader(path, encoding="utf-8") 
            pages.extend(loader.load())

        # แปลง Langchain Documents เป็น LlamaIndex Documents
        llama_docs = []
        for page in pages:
            llama_doc = Document(
                text=page.page_content,
                metadata=page.metadata
            )
            llama_docs.append(llama_doc)

        # ใช้ MarkdownNodeParser แบ่ง chunks
        chunks = self.md_parser.get_nodes_from_documents(llama_docs)

        print(f"Split documents into {len(chunks)} chunks")

        # แปลง LlamaIndex Nodes กลับเป็น Langchain Documents
        lc_chunks = []
        for chunk in chunks:
            lc_doc = LangchainDocument(
                page_content=chunk.text,
                metadata=chunk.metadata
            )
            lc_chunks.append(lc_doc)

        return lc_chunks

    def get_context_prompt(self, document_content, chunk_content):
        """สร้าง prompt สำหรับการสร้างบริบท (เหมือนในตัวอย่าง Anthropic)"""
        prompt = f"""<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval. Answer only with the succinct context and nothing else.
"""
        return prompt

    def generate_context(self, document_content, chunk_content):
        """สร้างบริบทโดยใช้ Qwen API"""
        prompt = self.get_context_prompt(document_content, chunk_content)
        context = self.call_qwen_api(prompt, max_tokens=100, temperature=0.1)
        return context.strip()

    def add_documents_with_context(self, docs):
        """เพิ่มเอกสารเข้า OpenSearch พร้อมบริบทเสริม (แบบที่ทำในตัวอย่าง)"""
        print(f"\nAdding {len(docs)} documents to OpenSearch with contextual information...")

        # 1. เตรียมข้อมูลสำหรับ vector index
        vector_bulk_data = []
        
        # 2. เตรียมข้อมูลสำหรับ BM25 index
        bm25_bulk_data = []

        for i, doc in enumerate(docs):
            # หาเอกสารเต็ม
            doc_source = doc.metadata.get("source", "")
            full_doc_content = doc.page_content  # ถ้าไม่มีเอกสารเต็ม ใช้ chunk เอง
            
            # สร้างบริบทสำหรับ chunk นี้
            context = self.generate_context(full_doc_content, doc.page_content)
            print(f"Generated context for chunk {i+1}/{len(docs)}: {context[:50]}...")
            
            # สร้าง contextualized content โดยรวมเนื้อหาและบริบทเข้าด้วยกัน (สำหรับ embedding)
            contextualized_content = f"{doc.page_content}\n\n{context}"
            
            # สร้าง vector embedding จาก contextualized content
            embedding = self.encoder.embed_query(contextualized_content)
            
            # เตรียมข้อมูลสำหรับ vector index
            vector_doc = {
                "index": {
                    "_index": self.vector_index_name,
                    "_id": f"doc_{i}"
                }
            }
            vector_data = {
                "embedding": embedding,
                "doc_id": f"doc_{i}",
                "content": doc.page_content,
                "chunk_id": i
            }
            vector_bulk_data.append(vector_doc)
            vector_bulk_data.append(vector_data)
            
            # เตรียมข้อมูลสำหรับ BM25 index
            bm25_doc = {
                "index": {
                    "_index": self.bm25_index_name,
                    "_id": f"doc_{i}"
                }
            }
            bm25_data = {
                "content": doc.page_content,
                "contextualized_content": context,  # เก็บแยกเหมือนตัวอย่าง
                "doc_id": f"doc_{i}",
                "chunk_id": i,
                "original_index": i
            }
            bm25_bulk_data.append(bm25_doc)
            bm25_bulk_data.append(bm25_data)

        # อัปโหลด vector data
        if vector_bulk_data:
            response = self.opensearch_client.bulk(body=vector_bulk_data)
            if response.get("errors", True):
                print("Some errors occurred during vector bulk upload.")
            else:
                print(f"Successfully added {len(docs)} documents to vector index")
            self.opensearch_client.indices.refresh(index=self.vector_index_name)
            
        # อัปโหลด BM25 data
        if bm25_bulk_data:
            response = self.opensearch_client.bulk(body=bm25_bulk_data)
            if response.get("errors", True):
                print("Some errors occurred during BM25 bulk upload.")
            else:
                print(f"Successfully added {len(docs)} documents to BM25 index")
            self.opensearch_client.indices.refresh(index=self.bm25_index_name)

        return len(docs)

    def sparse_search(self, query, k=10):
        """ค้นหาด้วย BM25 (คล้ายตัวอย่าง)"""
        response = self.opensearch_client.search(
            index=self.bm25_index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "contextualized_content"],
                        "type": "best_fields"
                    }
                },
                "size": k
            }
        )

        return [(hit["_id"], hit["_score"], hit["_source"]) for hit in response["hits"]["hits"]]

    def dense_search(self, query, k=10):
        """ค้นหาด้วย vector similarity"""
        query_embedding = self.encoder.embed_query(query)

        response = self.opensearch_client.search(
            index=self.vector_index_name,
            body={
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": k
                        }
                    }
                },
                "size": k
            }
        )

        return [(hit["_id"], hit["_score"], hit["_source"]) for hit in response["hits"]["hits"]]

    def rrf_fusion(self, rankings, k=60):
        """ผสมผลลัพธ์การค้นหาด้วย Reciprocal Rank Fusion (เหมือนในตัวอย่าง)"""
        fused_scores = {}
        doc_sources = {}

        for ranker_id, results in enumerate(rankings):
            for rank, (doc_id, score, source) in enumerate(results):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_sources[doc_id] = source

                # RRF formula: 1 / (k + rank)
                fused_scores[doc_id] += 1.0 / (k + rank)

        # เรียงลำดับตาม RRF score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_sources[doc_id]) for doc_id, score in sorted_results]

    def hybrid_search(self, query, k=5, rrf_k=60):
        """ค้นหาแบบ hybrid ด้วย RRF ที่รวมผลจาก sparse และ dense search (เหมือนตัวอย่าง)"""
        # ค้นหาด้วย BM25 (sparse)
        sparse_results = self.sparse_search(query, k=k*2)
        print(f"\nSparse search found {len(sparse_results)} results")

        # ค้นหาด้วย vector similarity (dense)
        dense_results = self.dense_search(query, k=k*2)
        print(f"Dense search found {len(dense_results)} results")

        # ใช้ RRF ผสมผลลัพธ์
        rankings = [sparse_results, dense_results]
        fused_results = self.rrf_fusion(rankings, k=rrf_k)
        print(f"RRF fusion returned {len(fused_results)} results")

        return fused_results[:k]

    def truncate_context(self, context, max_tokens=4000):
        """ย่อ context ให้มีขนาดไม่เกิน max_tokens"""
        words = context.split()
        estimated_tokens = len(words) * 1.3
        
        if estimated_tokens <= max_tokens:
            return context
        
        target_words = int(max_tokens / 1.3)
        truncated_context = " ".join(words[:target_words])
        return truncated_context + " ...(truncated)"

    def search_for_question(self, question, k=5):
        """ค้นหาเอกสารที่เกี่ยวข้องกับคำถามและสร้างคำตอบ"""
        results = self.hybrid_search(question, k=k)

        if not results:
            return self.generate_response(question)

        # แสดงเอกสารที่พบและใช้เนื้อหาเป็น context
        context = ""
        print(f"\nFound {len(results)} relevant documents for question: '{question}'")
        for i, (doc_id, score, doc) in enumerate(results, 1):
            print(f"\nDOCUMENT {i}/{len(results)}")
            print(f"Score: {score}")
            print(f"Doc ID: {doc_id}")

            # แสดงบริบทที่สร้างขึ้น
            if "contextualized_content" in doc:
                print(f"Context: {doc['contextualized_content']}")

            # แสดงเนื้อหาเฉพาะเอกสารแรก
            if i == 1:
                print("\nContent:")
                content = doc.get("content", "")
                if len(content) > 500:
                    print(content[:500] + "...(truncated)")
                else:
                    print(content)

            # เพิ่มเนื้อหาและบริบทเข้าไปใน context
            content = doc.get("content", "")
            context_info = doc.get("contextualized_content", "")
            context += f"Content: {content}\nContext: {context_info}\n\n"

        # ย่อ context ถ้าจำเป็น
        truncated_context = self.truncate_context(context)
        
        # สร้างคำตอบจาก context
        return self.generate_response(question, truncated_context)

    def generate_response(self, question, context="", max_tokens=1000):
        """สร้างคำตอบจากคำถามและ context"""
        if not context:
            prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Question: {question}"""
        else:
            prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Context information:
{context}

Question: {question}"""

        return self.call_qwen_api(prompt, max_tokens=max_tokens, temperature=0.6)

# ฟังก์ชันดาวน์โหลด corpus
def download_corpus():
    import urllib.request
    import os

    os.makedirs('./corpus_input', exist_ok=True)


# ฟังก์ชันหลักสำหรับรันระบบ
def main():
    # ตรวจสอบ API key
    if "DASHSCOPE_API_KEY" not in os.environ:
        print("Error: DASHSCOPE_API_KEY environment variable is not set")
        print("Please set it with: export DASHSCOPE_API_KEY='your_api_key_here'")
        return

    # ดาวน์โหลด corpus
    download_corpus()

    md_paths = [
        "./corpus_input/1.md",
        "./corpus_input/2.md",
        "./corpus_input/44.md",
        "./corpus_input/5555.md",   
    ]

    # ตั้งค่าระบบ Contextual RAG แบบ Anthropic
    rag = AnthropicStyleContextualRAG(
        opensearch_host="localhost",  # OpenSearch IP ตามที่ให้ไว้
        opensearch_port=9200
    )

    # โหลดเอกสาร
    docs = rag.load_documents(md_paths)
    print(f"Loaded {len(docs)} documents")

    # เพิ่มเอกสารเข้า OpenSearch พร้อมสร้างบริบท
    rag.add_documents_with_context(docs)

    # ทดสอบค้นหาและตอบคำถาม
    question = "ผมจะรู้ได้อย่างไรว่าผมเป็นโรคหัดแบบไหน"
    print(f"\nQuestion: {question}")

    answer = rag.search_for_question(question)
    print(f"\nFinal Answer: {answer}")

    # ทดสอบคำถามอีกข้อ
    question2 = "โรคต้อกระจก หรือโรคอะไรที่เกี่ยวข้องกันบ้าง"
    print(f"\nQuestion: {question2}")

    answer2 = rag.search_for_question(question2)
    print(f"\nFinal Answer: {answer2}")

if __name__ == "__main__":
    main()