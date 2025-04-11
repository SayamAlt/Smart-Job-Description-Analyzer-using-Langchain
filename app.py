from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import os, warnings, tempfile, re
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from io import BytesIO
from textwrap import wrap
from reportlab.lib.styles import ParagraphStyle
from re import sub

# Load environment variables
load_dotenv()

# Set OpenAI API key
if 'secrets' in st.secrets:
    api_key = st.secrets['secrets']['OPENAI_API_KEY']
elif 'OPENAI_API_KEY' in os.environ:
    api_key = os.getenv("OPENAI_API_KEY")
else:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

embeddings = OpenAIEmbeddings(api_key=api_key)
llm = OpenAI(api_key=api_key, temperature=0.5)
chat_model = ChatOpenAI(api_key=api_key, temperature=0.5)

# Define generic categories for skills
GENERIC_CATEGORIES = [
    "Programming",
    "Cloud Platforms",
    "Data & Analytics",
    "Machine Learning & AI",
    "DevOps & Tools",
    "Soft Skills",
    "Business & Strategy",
    "Finance & Accounting",
    "Marketing & CRM",
    "Project & Operations"
]


def extract_skills_and_qualifications(text):
    """
        Extracts skills and qualifications from the given text.
    """
    template = """
         You are a skilled resume and job description parser and analyzer. 
         Your task is to extract all relevant skills and qualifications from the given text.\n{text}
         \nReturn nothing else, just a comma-separated list of relevant skills and qualifications.
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["text"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    chain.output_parser = StrOutputParser()
    result = chain.invoke({"text": text})
    response = result["text"] if isinstance(result, dict) and "text" in result else str(result)
    return [skill.strip() for skill in response.split(",") if skill.strip()]

def filter_relevant_skills_and_qualifications(mixed_terms):
    """
        Filters out irrelevant skills and qualifications from the given text.
    """
    prompt = PromptTemplate(
        template="You are an expert HR assistant. " \
        "Your task is to filter out irrelevant skills and qualifications from the following text:\n{mixed_terms}" \
        "\nIgnore anything related to location, citizenship, job type (e.g., co-op/full time), team setup, duration, academic status, office names, or similar metadata." \
        "\nResponse with nothing else, just a comma-separated list of relevant skills and qualifications.",
        input_variables=["mixed_terms"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    chain.output_parser = StrOutputParser()
    result = chain.invoke({"mixed_terms": mixed_terms})
    response = result["text"] if isinstance(result, dict) and "text" in result else str(result)
    return [skill.strip() for skill in response.split(",") if skill.strip()]

def generate_rag_suggestions(missing_skills, vector_store):
    """
        Generate suggestions for resume improvement based on missing skills.
    """

    prompt = f"""
                You are an expert resume consultant. 
                Your task is to provide innovative and valuable suggestions for resume improvement 
                based on missing skills and qualifications which can help a candidate to stand out in the job market.
                Suggest insightful and informative tips for resume improvement to address the following missing skills:\n{missing_skills}
            """
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain.run(prompt)

def analyze_skill_gaps(jd_keywords, resume_keywords, threshold=60):
    """
    Match JD keywords to resume keywords using fuzzy matching.
    Returns a list of JD skills not found in resume.
    """
    matched = []

    for jd_kw in jd_keywords:
        jd_kw_lower = jd_kw.lower()

        for resume_kw in resume_keywords:
            score = fuzz.partial_ratio(jd_kw_lower, resume_kw.lower())

            if score >= threshold:
                matched.append(jd_kw)
                break

    missing = list(set(jd_keywords) - set(matched))
    return missing

def categorize_skills_and_qualifications(skills, categories):
    template = """
    You are an expert HR AI assistant. Given a list of general skill or qualification terms, 
    assign each to the most suitable category from the following:

    {categories}

    Respond with each skill in a new line using this format:
    Skill: Category

    Skills:
    {skills}
    """
    prompt = PromptTemplate(
        input_variables=["skills", "categories"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    skills_str = "\n".join(skills)
    categories_str = ", ".join(categories)

    response = chain.run({"skills": skills_str, "categories": categories_str})

    # Parse LLM response
    skill_category_map = {}

    for line in response.strip().split("\n"):
        if ":" in line:
            skill, category = line.split(":", 1)
            skill_category_map[skill.strip()] = category.strip()

    return skill_category_map

def group_skills_by_category(skill_category_map):
    grouped = defaultdict(list)

    # Group skills by category
    for skill, category in skill_category_map.items():
        grouped[category].append(skill)

    return dict(grouped)

def visualize_skill_match(jd_keywords, missing_skills):
    matched = list(set(jd_keywords) - set(missing_skills))
    unmatched = list(missing_skills)

    data = {
        'Skill': matched + unmatched,
        'Match': ['Matched'] * len(matched) + ['Missing'] * len(unmatched)
    }

    fig = px.bar(
        data,
        x='Skill',
        color='Match',
        title='Skill Match Analysis',
        color_discrete_map={'Matched': '#00cc96', 'Missing': '#EF553B'}
    )
    fig.update_layout(
        xaxis_title='Skills',
        yaxis_title='Count',
        barmode='group',
        xaxis_tickangle=-45
    )
    return fig

def visualize_skill_coverage(matched_count, missing_count):
    fig = go.Figure(data=[go.Pie(
        labels=['Matched Skills', 'Missing Skills'],
        values=[matched_count, missing_count],
        hole=0.4,  # for a donut chart effect
        marker=dict(colors=['#00cc96', '#EF553B'])
    )])

    fig.update_layout(title='Overall Skill Coverage')
    return fig

def visualize_match_score_distribution(jd_keywords, resume_keywords):
    scores = []

    # Calculate match scores for each JD keyword against resume keywords
    for jd_kw in jd_keywords:
        for resume_kw in resume_keywords:
            score = fuzz.partial_ratio(jd_kw.lower(), resume_kw.lower())
            if 0 <= score < 100:
                scores.append(score)

    fig = px.histogram(
        x=scores,
        nbins=10,
        title="Match Score Distribution (Resume vs JD)",
        labels={"x": "Resume vs JD Match Score", "y": "Frequency"},
        color_discrete_sequence=["#636EFA"]
    )

    fig.update_layout(xaxis=dict(dtick=10))
    return fig

def visualize_skill_category_distribution(skill_category_map):
    df = pd.DataFrame.from_dict(skill_category_map, orient='index', columns=['Matched Count'])
    df.reset_index(inplace=True)
    df.columns = ['Category', 'Matched Count']

    fig = px.line_polar(df, r='Matched Count', theta='Category', line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(title='Skill Category Strength Radar')
    return fig

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002500-\U00002BEF"  # Chinese characters
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def generate_pdf_report(missing_skills, score, feedback, suggestions, categorized_skills):
    feedback = remove_emojis(feedback)
    suggestions = remove_emojis(suggestions)
    missing_skills = [remove_emojis(skill) for skill in missing_skills]
    categorized_skills = {
        remove_emojis(cat): [remove_emojis(skill) for skill in skills]
        for cat, skills in categorized_skills.items()
    }
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=40, bottomMargin=40, rightMargin=50, leftMargin=50)

    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = styles["Heading2"]
    bullet = ParagraphStyle(name="Bullet", parent=normal, leftIndent=20, bulletIndent=10, spaceAfter=6)
    title = ParagraphStyle(name="Title", parent=normal, fontSize=16, alignment=TA_LEFT, spaceAfter=16, textColor=colors.HexColor('#333333'))

    elements = []

    # Title
    elements.append(Paragraph("<b>Smart Job Description Analyzer Report</b>", title))
    elements.append(Spacer(1, 12))

    # Match score
    elements.append(Paragraph(f"<b>Resume Match Score:</b> {score} / 10", normal))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Feedback:</b> {feedback}", normal))
    elements.append(Spacer(1, 12))

    # Missing Skills
    elements.append(Paragraph("<b>Missing Skills</b>", heading))
    if missing_skills:
        for skill in missing_skills:
            elements.append(Paragraph(f"• {skill}", bullet))
    else:
        elements.append(Paragraph("None – great coverage!", normal))

    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Suggestions for Resume Improvement</b>", heading))

    for line in suggestions.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Replace **bold** markers with <b> HTML tags
        line = sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)

        # Add spacing for bullet points and headers
        if line.startswith("-"):
            elements.append(Paragraph(line, bullet))
        elif line[0].isdigit() and line[1:3] in [". ", ".)"]:  # numbered point
            elements.append(Paragraph(f"<b>{line}</b>", normal))
        else:
            elements.append(Paragraph(line, normal))

    # Categorized Skills
    elements.append(Paragraph("<b>Categorized Resume Skills</b>", heading))

    for category, skills in categorized_skills.items():
        skill_str = ", ".join(skills)
        if skill_str:
            elements.append(Paragraph(f"<b>{category}:</b> {skill_str}", normal))

    elements.append(Spacer(1, 12))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Streamlit UI
st.set_page_config(page_title="Smart Job Description Analyzer", layout="centered")
st.title("📊 Smart Job Description Analyzer")

st.markdown("### 🧾 Enter the Job Description")
jd_text = st.text_area(" ", height=250)

st.markdown("### 📄 Upload Your Resume (PDF)")
resume_file = st.file_uploader(" ", type=["pdf"])

if st.button("🔍 Analyze"): 
    if jd_text and resume_file:
        with st.spinner("🧠 Analyzing..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(resume_file.read())
                tmp_path = tmp_file.name

            loader = PDFPlumberLoader(tmp_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splitted_docs = text_splitter.split_documents(documents)

            content = "\n".join([doc.page_content for doc in splitted_docs])

            # Create FAISS vector store from resume
            faiss = FAISS.from_documents(splitted_docs, embeddings)

            resume_keywords = extract_skills_and_qualifications(content)
            jd_keywords = extract_skills_and_qualifications(jd_text)

            # Filter out irrelevant skills and qualifications
            jd_relevant_keywords = filter_relevant_skills_and_qualifications(jd_keywords)
            resume_relevant_keywords = filter_relevant_skills_and_qualifications(resume_keywords)

            # Categorize skills and qualifications
            skill_category_map = categorize_skills_and_qualifications(resume_relevant_keywords, GENERIC_CATEGORIES)
            categorized_skills = group_skills_by_category(skill_category_map)

            # Count the number of skills in each category
            num_skills_by_category = {category: len(skills) for category, skills in categorized_skills.items()}

            # Analyze skill gaps
            missing_skills = analyze_skill_gaps(jd_relevant_keywords, resume_relevant_keywords)

            # Calculate match score based on the number of matched keywords
            matched = len(jd_relevant_keywords) - len(missing_skills)
            total = len(jd_relevant_keywords)
            score = round((matched / total) * 10, 1) if total > 0 else 0.0

            # Feedback based on score
            if score >= 9:
                feedback = "🌟 Excellent match! Your resume aligns very closely with the job description."
            elif score >= 7:
                feedback = "✅ Good match. You meet most of the key requirements. A few improvements can strengthen it."
            elif score >= 4:
                feedback = "⚠️ Fair match. Several important qualifications seem to be missing. Consider revising your resume."
            else:
                feedback = "❌ Poor match. Your resume covers few of the listed requirements. Major updates recommended."

            # RAG-powered suggestions
            suggestions = generate_rag_suggestions(missing_skills, faiss)

            st.success("✅ Analysis Complete!")

            st.subheader("🧠 Keywords Found")
            st.write(f"Job Description Keywords: {', '.join(jd_relevant_keywords)}")
            st.write(f"Resume Keywords: {', '.join(resume_relevant_keywords)}")

            st.subheader("🚫 Missing Skills & Qualifications in Resume")
            st.write(missing_skills if missing_skills else "None – great coverage!")

            st.subheader("🎯 Resume Match Score")
            st.metric(label="Match Score", value=f"{score} / 10")
            st.write(feedback)

            st.subheader("📊 Skill Match Bar Chart")
            st.plotly_chart(visualize_skill_match(jd_relevant_keywords, missing_skills), use_container_width=True)

            st.subheader("🥧 Overall Skill Coverage (Donut Chart)")
            st.plotly_chart(visualize_skill_coverage(matched, len(missing_skills)), use_container_width=True)

            st.subheader("📡 Skill Category Strength (Radar View)")
            st.plotly_chart(visualize_skill_category_distribution(num_skills_by_category), use_container_width=True)

            st.subheader("📈 Resume Match Score Distribution")
            st.plotly_chart(visualize_match_score_distribution(jd_relevant_keywords, resume_relevant_keywords))

            st.subheader("💡 Suggestions for Resume Improvement")
            st.info(suggestions)

            # PDF Download Button
            pdf_bytes = generate_pdf_report(
                missing_skills,
                score,
                feedback,
                suggestions,
                categorized_skills
            )

            col1, col2, col3 = st.columns([2, 3, 2])
            
            with col2:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name="JD_Resume_Analysis_Report.pdf",
                    mime="application/pdf"
                )
    elif jd_text or resume_file:
        if not jd_text:
            st.warning("Please enter a job description.")
        elif not resume_file:
            st.warning("Please upload your resume in PDF format.")


