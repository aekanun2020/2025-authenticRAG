# Search-only Authentic RAG

ระบบค้นหาข้อมูลแบบ Hybrid (BM25 + Vector) พร้อม RRF Fusion สำหรับค้นหาจาก indexed documents

## Overview

`onlysearchAuthenticRAG.py` เป็นส่วนค้นหาที่แยกออกมาจาก `authenticRAG.py` ใช้สำหรับ:
- ค้นหาข้อมูลจาก documents ที่ index แล้ว
- ประมวลผลหลายคำถามพร้อมกัน (batch processing)
- Export ผลลัพธ์เป็น JSON

## การทำงานของ Retriever

### 1. Sparse Search (BM25)

#### การค้นหา:
```python
def sparse_search(self, query, k=10):
    response = self.opensearch_client.search(
        index=self.bm25_index_name,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                    "type": "best_fields"
                }
            }
        }
    )
```

#### Index ที่ใช้: `anthropic-bm25-index`

#### ข้อมูลที่เก็บ:
```json
{
    "content": "ภาวะแทรกซ้อนที่พบได้ในหญิงตั้งครรภ์...",
    "contextualized_content": "บทความนี้กล่าวถึงภาวะแทรกซ้อนของโรคหัดเยอรมัน...",
    "doc_id": "doc_6",
    "chunk_id": 6
}
```

#### การทำงานของ BM25:

**Indexing Phase (ตอนเก็บข้อมูล):**
1. OpenSearch รับ text จาก fields `content` และ `contextualized_content`
2. ใช้ Standard Analyzer แยกคำ (tokenization)
3. สร้าง inverted index เก็บ:
   - Term positions (คำอยู่ตำแหน่งไหน)
   - Term frequency (คำแต่ละคำปรากฏกี่ครั้ง)
   - Document frequency (คำนั้นอยู่ในกี่ documents)
   - Document length statistics
4. **ยังไม่คำนวณ BM25 score** - เก็บแค่ statistics

**Search Phase (ตอนค้นหา):**
1. รับ query แล้วแยกคำเหมือนกัน
2. ดึง statistics ของ matching terms จาก index
3. **คำนวณ BM25 score แบบ real-time** จาก statistics ที่เก็บไว้
4. Return documents เรียงตาม score

#### BM25 Scoring:
```
Score = Σ IDF(term) × TF_score(term, doc)

โดยที่:
- IDF = log((N-df+0.5)/(df+0.5))
- TF_score คำนวณจาก term frequency และ document length normalization
```

### 2. Dense Search (Vector)

#### การค้นหา:
```python
def dense_search(self, query, k=10):
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
            }
        }
    )
```

#### Index ที่ใช้: `anthropic-vector-index`

#### ข้อมูลที่เก็บ:
- `embedding`: vector 1024 มิติ (จาก BAAI/bge-m3)
- `content`: เนื้อหา chunk
- `doc_id`, `chunk_id`

#### Vector Scoring:
- ใช้ cosine similarity
- Score range: 0-1 (1 = เหมือนกันสมบูรณ์)

### 3. Hybrid Search with RRF

#### การทำงาน:
```python
def hybrid_search(self, query, k=5, rrf_k=60):
    # 1. ค้นหา 2 วิธี
    sparse_results = self.sparse_search(query, k=k*2)  # 10 docs
    dense_results = self.dense_search(query, k=k*2)    # 10 docs
    
    # 2. RRF fusion
    rankings = [sparse_results, dense_results]
    fused_results = self.rrf_fusion(rankings, k=rrf_k)
    
    # 3. คืนค่า top-k
    return fused_results[:k]  # 5 docs
```

### 4. RRF (Reciprocal Rank Fusion)

#### สูตร:
```python
RRF_score(doc) = Σ 1/(k + rank)
```

#### ตัวอย่างการคำนวณ:

| Document | BM25 Rank | Vector Rank | RRF Calculation | Final Score |
|----------|-----------|-------------|-----------------|-------------|
| doc_A | 1 (rank=0) | 3 (rank=2) | 1/60 + 1/62 | 0.0328 |
| doc_B | 2 (rank=1) | - | 1/61 + 0 | 0.0164 |
| doc_C | - | 1 (rank=0) | 0 + 1/60 | 0.0167 |

### 5. Response Generation

หลังจากได้ top-k documents:

```python
def search_for_question(self, question, k=5):
    # 1. Hybrid search
    results = self.hybrid_search(question, k=k)
    
    # 2. สร้าง context จาก results
    context = ""
    for doc in results:
        context += f"Content: {doc['content']}\n"
        context += f"Context: {doc['contextualized_content']}\n\n"
    
    # 3. Generate answer
    answer = self.generate_response(question, context)
    
    return {
        "question": question,
        "answer": answer,
        "results": search_results
    }
```

## ตัวอย่างการใช้งาน

### Single Question:
```python
rag = AuthenticSearchRAG()
result = rag.search_for_question("อันตรายของหัดเยอรมันกับหญิงตั้งครรภ์")
print(result["answer"])
```

### Multiple Questions:
```python
questions = [
    "โรคหัดและโรคหัดเยอรมันต่างกันอย่างไร?",
    "วิธีป้องกันโรคหัดเยอรมัน"
]
results = rag.search_multiple_questions(questions, k=5)
rag.export_results_to_json(results, "output.json")
```

## Output Format

```json
{
    "question": "โรคหัดและโรคหัดเยอรมันต่างกันอย่างไร?",
    "answer": "โรคหัดและโรคหัดเยอรมันเป็นโรคคนละชนิด...",
    "results": [
        {
            "doc_id": "doc_3",
            "score": 0.0328,
            "content": "โรคหัดเยอรมัน (Rubella)...",
            "context": "บทความนี้อธิบายความแตกต่าง..."
        }
    ]
}
```

## Performance Considerations

### Search Performance:
- BM25: O(log n) per query term
- Vector: O(n) for brute force k-NN
- RRF: O(k log k) for sorting

### Quality Trade-offs:
- `k=5`: เร็ว แต่อาจพลาดข้อมูลสำคัญ
- `k=10`: สมดุลระหว่าง speed และ recall
- `k=20`: ครอบคลุม แต่ช้าและ context ยาว

## Requirements
- OpenSearch ต้อง running และมีข้อมูล indexed แล้ว
- ใช้ indices เดียวกับ authenticRAG.py
- DASHSCOPE_API_KEY สำหรับ answer generation