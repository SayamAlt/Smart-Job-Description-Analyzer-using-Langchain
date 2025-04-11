# 📊 Smart Job Description Analyzer

The **Smart Job Description Analyzer** is a powerful LLM-based application that helps users intelligently evaluate how well their resumes match a given job description. It provides actionable insights, visual feedback, and auto-generated suggestions to help improve resume alignment and job fit.

This project showcases advanced applications of **LangChain**, **OpenAI GPT-4**, **FAISS**, and **Streamlit**, along with semantic skill categorization, fuzzy matching, and PDF report generation for a complete end-to-end experience.

---

## 🚀 Features

- 🔍 **Job Description Analysis** – Extracts relevant skills, qualifications, and expectations using LLMs.
- 📄 **Resume Parsing** – Processes resume PDFs and retrieves core skills using chunked embedding.
- 🧠 **Skill Gap Detection** – Uses fuzzy matching to detect unmatched skills between JD and resume.
- 🧠 **AI Suggestions** – Provides resume improvement tips powered by Retrieval-Augmented Generation (RAG).
- 📌 **Resume Match Score** – Computes a job fit score out of 10 with personalized feedback.
- 🗂️ **Auto-Categorized Skills** – Dynamically categorizes resume skills into general domains (e.g., Programming, Marketing, Cloud).
- 📊 **Visual Insights** – Includes interactive bar charts and radar plots for skill match analysis.
- 📄 **PDF Report Export** – Download a cleanly formatted resume-JD analysis report.
- ✅ **Fuzzy Matching** – Uses `fuzzywuzzy` for partial match detection, enabling more accurate comparisons.

---

## 🧠 Technologies Used

| Layer             | Stack                          |
|------------------|---------------------------------|
| LLMs              | OpenAI GPT-4 via LangChain     |
| Embeddings        | OpenAI `text-embedding-ada-002`|
| Vector Store      | FAISS                          |
| Frontend          | Streamlit                      |
| Visualization     | Plotly                         |
| Parsing & Utils   | PDFPlumber, ReportLab, Regex   |
| NLP & Matching    | FuzzyWuzzy                     |

---

## 🧱 Architecture

1. User inputs a job description (as text).
2. Uploads a resume (PDF).
3. JD and Resume are parsed and embedded.
4. LangChain is used to extract and categorize skills.
5. FAISS enables similarity comparison.
6. Match score, gaps, and LLM-generated suggestions are calculated.
7. Charts and download options are rendered via Streamlit.

---

## 📸 Screenshots

> Add screenshots here once deployed (e.g., of match score, radar chart, PDF report)

---

## 🧪 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/smart-jd-analyzer.git
cd smart-jd-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a .env file with:

```bash
OPENAI_API_KEY=your_openai_api_key
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🧾 Sample Use Cases

<ul>
    <li>✅ Job seekers tailoring resumes for better ATS and recruiter screening</li>
    <li>✅ Recruiters checking candidate relevance to job postings</li>
    <li>✅ Career counselors assisting clients with resume optimization</li>
    <li>✅ HR tech innovators building smarter hiring tools</li>
</ul>

--- 

## 📄 PDF Report Includes:

<ol>
    <li>Match Score and Feedback</li>
    <li>Missing Skills & Improvement Suggestions</li>
    <li>Categorized Skills by Domain</li>   
    <li>Clean, wrapped, emoji-free formatting</li>
</ol>

---

## 💡 Future Improvements

<ul>
    <li>Add chatbot assistant for interactive guidance</li>
    <li>Support for multiple resume comparison</li>
    <li>Integration with ATS or job boards</li>
    <li>Historical tracking of resume improvements</li>
</ul>

---

## 🤝 Contributing

Pull requests and suggestions are welcome! For major changes, please open an issue first to discuss what you would like to change.