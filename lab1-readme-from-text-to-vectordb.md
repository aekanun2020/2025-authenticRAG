# Anthropic-style Contextual RAG with Qwen API

## Overview
ระบบ RAG (Retrieval Augmented Generation) แบบ Contextual ที่พัฒนาตามแนวทางของ Anthropic โดยใช้ Qwen API แทน Claude API

## Key Features
- **Contextual Embeddings**: สร้าง context สำหรับแต่ละ chunk เพื่อเพิ่มประสิทธิภาพการค้นหา
- **Hybrid Search**: ใช้ทั้ง BM25 (sparse) และ Vector (dense) search
- **OpenSearch Integration**: ใช้ OpenSearch เป็น vector database

## Requirements
```bash
pip install opensearchpy sentence-transformers llama-index llama-index-embeddings-huggingface openai
```

## Setup

### 1. ติดตั้ง OpenSearch
```bash
# ใช้ Docker (แนะนำ)
docker run -d -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  opensearchproject/opensearch:latest
```

### 2. ตั้งค่า API Key
```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 3. เตรียมเอกสาร
วางไฟล์ markdown ใน folder `./corpus_input/`:
```
corpus_input/
├── 1.md
├── 2.md
├── 44.md
└── 5555.md
```

## Code Walkthrough: การนำเข้าข้อมูลสู่ Vector Database

### 1. การสร้าง Indices ใน OpenSearch
```python
def _create_or_update_indices(self):
    # Vector index สำหรับ semantic search
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
                }
            }
        }
    }
    
    # BM25 index สำหรับ keyword search
    bm25_settings = {
        "settings": {
            "similarity": {"default": {"type": "BM25"}}
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "contextualized_content": {"type": "text"}
            }
        }
    }
```

### 2. การสร้าง Context สำหรับแต่ละ Chunk
```python
def get_context_prompt(self, document_content, chunk_content):
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
```

### 3. การเตรียมข้อมูลและ Indexing
```python
def add_documents_with_context(self, docs):
    for i, doc in enumerate(docs):
        # สร้าง context สำหรับ chunk
        context = self.generate_context(full_doc_content, doc.page_content)
        
        # สำหรับ Vector Index: embed ทั้ง content + context
        contextualized_content = f"{doc.page_content}\n\n{context}"
        embedding = self.encoder.embed_query(contextualized_content)
        
        # Vector data
        vector_data = {
            "embedding": embedding,
            "doc_id": f"doc_{i}",
            "content": doc.page_content
        }
        
        # BM25 data - เก็บแยก content กับ context
        bm25_data = {
            "content": doc.page_content,
            "contextualized_content": context,
            "doc_id": f"doc_{i}"
        }
```

### 4. Bulk Upload ไปยัง OpenSearch
```python
# Upload vector embeddings
response = self.opensearch_client.bulk(body=vector_bulk_data)

# Upload BM25 text data  
response = self.opensearch_client.bulk(body=bm25_bulk_data)

# Refresh indices เพื่อให้ searchable ทันที
self.opensearch_client.indices.refresh(index=self.vector_index_name)
self.opensearch_client.indices.refresh(index=self.bm25_index_name)
```

## การทำงานของระบบ

### 1. Document Processing
- ใช้ TextLoader อ่านไฟล์ .md
- แปลงเป็น LlamaIndex Documents
- แบ่งด้วย MarkdownNodeParser(chunk_size=256)
- แปลงกลับเป็น Langchain Documents

### 2. Context Generation  
```
<document> 
{{WHOLE_DOCUMENT}} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{{CHUNK_CONTENT}} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
```

### 3. Dual Indexing
- **Vector Index**: เก็บ embedding ของ (chunk + context)
- **BM25 Index**: เก็บ text แยก content และ context

### 4. Document Flow
```
Input Documents → Split into Chunks → Generate Context for Each Chunk
                                                ↓
                                    Store in OpenSearch Indices:
                                    ├─ Vector Index (for semantic search)
                                    └─ BM25 Index (for keyword search)
```

## ตัวอย่างการใช้งาน

```python
```python
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
        opensearch_host="localhost",
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

if __name__ == "__main__":
    main()
```
```

## การ Debug

### ตรวจสอบ OpenSearch

# ดู indices ทั้งหมด
curl -X GET "localhost:9200/_cat/indices?v"

# ลบข้อมูลเก่า
curl -X POST "localhost:9200/anthropic-*/_delete_by_query?pretty" \
  -H 'Content-Type: application/json' \
  -d '{"query": {"match_all": {}}}'
```

### ปัญหาที่พบบ่อย

1. **Connection Refused**
   - ตรวจสอบว่า OpenSearch รันอยู่
   - ตรวจสอบ port 9200

2. **Invalid API Key**
   - ตรวจสอบ environment variable
   - สร้าง API key ใหม่หากจำเป็น

### 3. **No Search Results**
   - ตรวจสอบว่าคำถามสอดคล้องกับเอกสาร
   - BM25 ต้องการคำที่ตรงกันแบบ exact match

## Performance Considerations

### Latency
- Context generation: ~0.5-1s ต่อ chunk
- Search: ~100-200ms
- Answer generation: ~1-2s

### Cost
- ใช้ LLM 2 ครั้ง: สร้าง context (ตอน indexing) + ตอบคำถาม (ตอน search)
- Context generation: ~100-500 tokens ต่อ chunk
- Answer generation: ~1000-2000 tokens ต่อคำตอบ

### Storage
- Vector index: ~1.5MB ต่อ 50 chunks
- BM25 index: ~300KB ต่อ 50 chunks

## ข้อแตกต่างจาก Generic RAG

| Generic RAG | Contextual RAG |
|------------|----------------|
| Chunk → Embed → Store | Chunk → Generate Context → Embed(Chunk+Context) → Store |
| Single search method | Hybrid search (BM25 + Vector) |
| 1x LLM call | 2x LLM calls |

## Tips

1. **Context Quality**: ใช้ model ที่ดีสำหรับสร้าง context (temperature=0.1)
2. **Chunk Size**: 256 chars เหมาะสำหรับภาษาไทย
3. **RRF k parameter**: ค่า default 60 ทำงานได้ดีในหลายกรณี
4. **Document Matching**: ให้คำถามและเอกสารอยู่ในหัวข้อเดียวกัน

## References
- [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [OpenSearch Documentation](https://opensearch.org/docs/latest/)
- [Qwen API Documentation](https://help.aliyun.com/document_detail/2712195.html)