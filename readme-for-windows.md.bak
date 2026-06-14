# คู่มือการจัดเตรียมระบบ Authentic RAG สำหรับ Windows

คู่มือฉบับนี้จะช่วยให้คุณติดตั้งและใช้งานระบบ Authentic RAG บน Windows ได้อย่างง่ายดาย

## ความต้องการของระบบ

- Windows 10/11
- Python 3.10 ขึ้นไป
- Docker Desktop for Windows
- RAM 8GB ขึ้นไป (แนะนำ)
- API Key สำหรับ DashScope (Qwen API)
- Ollama

## โครงสร้างโฟลเดอร์

```
project_root/
├── corpus_input/          # โฟลเดอร์สำหรับเก็บไฟล์ Markdown
├── authenticRAG.py        # โค้ดหลักสำหรับ AuthenticRAG
├── onlysearchAuthenticRAG.py  # สคริปต์สำหรับการค้นหา
└── authentic_rag_search_results.json # ไฟล์ผลลัพธ์
```

## ขั้นตอนการติดตั้ง

### 0. ดาวน์โหลดและแตกไฟล์โปรเจกต์

**วิธีที่ 1: ดาวน์โหลดผ่านเว็บเบราว์เซอร์ (ง่ายที่สุด)**

1. ไปที่ https://github.com/aekanun2020/2025-authenticRAG
2. คลิกปุ่มสีเขียว **Code** → เลือก **Download ZIP**
3. บันทึกไฟล์ `2025-authenticRAG-main.zip` ลงในเครื่อง
4. คลิกขวาที่ไฟล์ → เลือก **Extract All...** → เลือกตำแหน่งที่ต้องการ → คลิก **Extract**
5. เปลี่ยนชื่อโฟลเดอร์จาก `2025-authenticRAG-main` เป็น `authenticRAG` (ถ้าต้องการ)

**วิธีที่ 2: ใช้ Git (สำหรับผู้ที่มี Git ติดตั้งอยู่แล้ว)**

เปิด Command Prompt หรือ PowerShell แล้วรัน:

```bash
git clone https://github.com/aekanun2020/2025-authenticRAG.git
cd 2025-authenticRAG
```

**วิธีที่ 3: ดาวน์โหลดด้วย PowerShell**

เปิด PowerShell และรันคำสั่ง:

```powershell
# ดาวน์โหลดไฟล์ ZIP
Invoke-WebRequest -Uri "https://github.com/aekanun2020/2025-authenticRAG/archive/refs/heads/main.zip" -OutFile "authenticRAG.zip"

# แตกไฟล์ ZIP
Expand-Archive -Path "authenticRAG.zip" -DestinationPath "." -Force

# เปลี่ยนชื่อโฟลเดอร์และเข้าไปในโฟลเดอร์
Rename-Item -Path "2025-authenticRAG-main" -NewName "authenticRAG"
cd authenticRAG
```

**หลังจากแตกไฟล์แล้ว** ให้เปิด Command Prompt หรือ PowerShell และเข้าไปในโฟลเดอร์โปรเจกต์:

```bash
cd C:\path\to\authenticRAG
```

### 1. สร้างสภาพแวดล้อมด้วย Conda

**หากยังไม่มี Conda:** ดาวน์โหลดและติดตั้ง Miniconda จาก https://www.anaconda.com/docs/getting-started/miniconda/install

เปิด **Anaconda Prompt** หรือ **Command Prompt** แล้วรันคำสั่ง:

```bash
conda create -n advrag python=3.10
conda activate advrag
```

### 2. ติดตั้ง Python Packages

```bash
pip install opensearch-py sentence-transformers llama-index llama-index-embeddings-huggingface openai tqdm
pip install langchain
pip install langchain_community
```

### 3. ติดตั้งและรัน OpenSearch ด้วย Docker

**หมายเหตุ:** ตรวจสอบว่าเปิด Docker Desktop แล้ว

เปิด **PowerShell** หรือ **Command Prompt** แล้วรันคำสั่ง:

```powershell
docker network create opensearch-net

docker run -d --name opensearch -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" -e "network.host=0.0.0.0" opensearchproject/opensearch:2.19.1
```

### 4. ตั้งค่า API Key สำหรับ DashScope

**ใน Command Prompt:**
```cmd
set DASHSCOPE_API_KEY=your_api_key_here
```

**ใน PowerShell:**
```powershell
$env:DASHSCOPE_API_KEY="your_api_key_here"
```

**เพื่อให้ API Key คงอยู่ถาวร:** ไปที่ System Properties → Environment Variables → เพิ่ม User Variable ชื่อ `DASHSCOPE_API_KEY` พร้อมค่า API Key ของคุณ

### 5. ติดตั้ง Ollama และโมเดล BGE-M3

1. ดาวน์โหลด Ollama สำหรับ Windows จาก https://ollama.com/download
2. ติดตั้งและเปิดโปรแกรม
3. เปิด Command Prompt หรือ PowerShell แล้วรัน:

```bash
ollama pull bge-m3
```

### 6. นำเข้าเอกสารเข้าสู่ระบบ

วางไฟล์ Markdown ของคุณในโฟลเดอร์ `corpus_input/` แล้วรัน:

```bash
python authenticRAG.py
```

**ต้องการปรับแต่ง?** ดูรายละเอียดที่: [lab1-readme-from-text-to-vectordb.md](lab1-readme-from-text-to-vectordb.md)

### 7. ค้นหาและวิเคราะห์เอกสาร

```bash
python onlysearchAuthenticRAG.py
```

**อ่านเพิ่มเติม:** [lab2-readme-from-vectordb-to-final-answer.md](lab2-readme-from-vectordb-to-final-answer.md)

### 8. แก้ไขปัญหา Error ของ langchain

หากพบปัญหาเกี่ยวกับ langchain ให้รัน:

```bash
pip uninstall langchain langchain-core langchain-community -y
pip install langchain==0.0.354
```

## สรุปคำสั่งสำคัญ

| ขั้นตอน | คำสั่ง Windows |
|---------|----------------|
| แตกไฟล์ ZIP | คลิกขวา → Extract All หรือใช้ `Expand-Archive` |
| เข้าโฟลเดอร์โปรเจกต์ | `cd C:\path\to\authenticRAG` |
| ตั้งค่า Environment Variable (ถาวร) | ตั้งค่าผ่าน System Properties → Environment Variables |
| ตั้งค่า API Key ชั่วคราว (CMD) | `set DASHSCOPE_API_KEY=your_key` |
| ตั้งค่า API Key ชั่วคราว (PowerShell) | `$env:DASHSCOPE_API_KEY="your_key"` |
| เช็ค Docker Network | `docker network ls` |
| เช็คสถานะ OpenSearch | `docker ps` |

## เคล็ดลับสำหรับผู้ใช้ Windows

- **ตรวจสอบโฟลเดอร์ที่แตกไฟล์:** หลังแตก ZIP ให้แน่ใจว่าเข้าไปในโฟลเดอร์ที่มีไฟล์ `authenticRAG.py`
- ใช้ **Anaconda Prompt** สำหรับคำสั่ง Python และ Conda
- ใช้ **PowerShell** หรือ **Command Prompt** สำหรับคำสั่ง Docker
- หากใช้ PowerShell อาจต้องเปิดใช้งาน execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- ตรวจสอบว่า Docker Desktop รันอยู่ก่อนใช้คำสั่ง Docker ทุกครั้ง
- หาก Windows Defender หรือ Antivirus บล็อกการแตกไฟล์ ให้อนุญาตชั่วคราว