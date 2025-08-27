# คู่มือการติดตั้งและใช้งานระบบ RAG ขั้นสูง

คู่มือนี้ให้คำแนะนำขั้นตอนต่อขั้นตอนสำหรับการติดตั้งและใช้งานระบบ RAG (Retrieval-Augmented Generation) ขั้นสูงด้วย OpenSearch, Ollama และ Python

## เริ่มต้นอย่างรวดเร็ว (Quick Start)

```bash
# 1. ติดตั้ง dependencies
pip install opensearchpy sentence-transformers llama-index llama-index-embeddings-huggingface openai tqdm

pip install langchain

pip install langchain_community


# 2. ติดตั้ง Docker และเปิด OpenSearch
docker network create opensearch-net
docker run -d \
  --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -e "network.host=0.0.0.0" \
  opensearchproject/opensearch:2.19.1


# 3. ตั้งค่า API Key สำหรับ DashScope (Qwen API)
export DASHSCOPE_API_KEY='your_api_key_here'

# 4. รันโค้ดสร้าง RAG System
python authenticRAG.py
```

## ความต้องการของระบบ

- Python 3.10+
- Docker Desktop
- RAM 8GB+ (แนะนำ)
- API Key สำหรับ DashScope (Qwen API)


## โครงสร้างโฟลเดอร์

```
project_root/
├── corpus_input/          # โฟลเดอร์สำหรับเก็บไฟล์ Markdown
├── authenticRAG.py        # โค้ดหลักสำหรับ AuthenticRAG
├── onlysearchAuthenticRAG.py  # สคริปต์สำหรับการค้นหาโดยใช้ AuthenticRAG
└── authentic_rag_search_results.json # ไฟล์ผลลัพธ์จากการค้นหา
```

## การแก้ไขปัญหา

### ปัญหาการเชื่อมต่อกับ OpenSearch

```bash
# ตรวจสอบว่า OpenSearch กำลังทำงานอยู่
curl http://localhost:9200

# เอาต์พุตที่คาดหวังควรมีข้อมูลเวอร์ชันของ OpenSearch
```

### ตรวจสอบสถานะ Docker

```bash
# ตรวจสอบสถานะคอนเทนเนอร์ Docker
docker ps -a | grep opensearch

# รีสตาร์ท OpenSearch หากจำเป็น
docker restart opensearch-single-node
```

### ปัญหาทั่วไปเกี่ยวกับ Python

- หากพบข้อผิดพลาดการนำเข้าโมดูล ตรวจสอบให้แน่ใจว่าสภาพแวดล้อมเสมือนของคุณเปิดใช้งานอยู่
- สำหรับข้อผิดพลาดที่เกี่ยวข้องกับ API Key ตรวจสอบให้แน่ใจว่าคุณได้ตั้งค่าตัวแปรสภาพแวดล้อม DASHSCOPE_API_KEY แล้ว

## การปรับแต่งเพิ่มเติม

### การใช้โมเดล Embedding อื่น

ระบบใช้โมเดล BAAI/bge-m3 เป็นค่าเริ่มต้น หากต้องการเปลี่ยนโมเดล ให้แก้ไขในโค้ด:

```python
# เปลี่ยนจาก
self.embed_model = SentenceTransformer('BAAI/bge-m3')

# เป็นโมเดลอื่น เช่น
self.embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
```

