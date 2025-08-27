# คู่มือการจัดเตรียมระบบ Authentic RAG

คู่มือนี้ให้คำแนะนำการจัดเตรียมระบบ ก่อนรัน authenticRAG.py หรือ onlysearchAuthenticRAG.py 

## ความต้องการของระบบ

- Python 3.10+
- Docker Desktop
- RAM 8GB+ (แนะนำ)
- API Key สำหรับ DashScope (Qwen API)
- Ollama


## โครงสร้างโฟลเดอร์

```
project_root/
├── corpus_input/          # โฟลเดอร์สำหรับเก็บไฟล์ Markdown
├── authenticRAG.py        # โค้ดหลักสำหรับ AuthenticRAG
├── onlysearchAuthenticRAG.py  # สคริปต์สำหรับการค้นหาโดยใช้ AuthenticRAG
└── authentic_rag_search_results.json # ไฟล์ผลลัพธ์จากการค้นหา
```


## การติดตั้งเพื่อจัดเตรียมระบบ

```bash

# 1. สร้างสภาพแวดล้อมใหม่ด้วย Conda
conda create -n advrag python=3.10
conda activate advrag

# 2. ติดตั้ง dependencies
pip install opensearchpy sentence-transformers llama-index llama-index-embeddings-huggingface openai tqdm

pip install langchain

pip install langchain_community


# 3. ติดตั้ง OpenSearch
docker network create opensearch-net
docker run -d \
  --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -e "network.host=0.0.0.0" \
  opensearchproject/opensearch:2.19.1


# 4. ตั้งค่า API Key สำหรับ DashScope (Qwen API)
export DASHSCOPE_API_KEY='your_api_key_here'

# 5. ติดตั้ง Ollama ตาม https://ollama.com/download แล้วดาวน์โหลด BAAI/bge-m3 ด้วยคำสั่ง
ollama pull bge-m3

