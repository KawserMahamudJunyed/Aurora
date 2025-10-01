import gradio as gr
import PyPDF2
import os
import json
import time
import google.generativeai as genai
import logging
from google.api_core import retry

# --- Setup Logging ---
logging.basicConfig(level=logging.ERROR)

# --- Configure Google AI Studio API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro-latest')

# --- Utility Functions ---
def validate_input(text, field_name, max_length=500):
    if not text or text.strip() == "":
        raise ValueError(f"{field_name} cannot be empty.")
    if len(text) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length of {max_length} characters.")
    return text.strip()

def safe_parse_json(response_text):
    try:
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json') and cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[7:-3].strip()
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return {}

@retry.Retry(predicate=retry.if_exception_type(Exception), initial=1.0, maximum=60.0, multiplier=2.0)
def call_gemini_api(prompt):
    return model.generate_content(prompt)

# --- Worker Functions ---
def analyze_cv(cv_file, job_description, progress=gr.Progress(track_tqdm=True)):
    # Initialize a default output dictionary for all components
    default_output = {
        'report_header': gr.update(visible=False),
        'matched_tech_header': gr.update(visible=False),
        'matched_tech_output': gr.update(visible=False),
        'matched_soft_header': gr.update(visible=False),
        'matched_soft_output': gr.update(visible=False),
        'missing_tech_header': gr.update(visible=False),
        'missing_tech_output': gr.update(visible=False),
        'missing_soft_header': gr.update(visible=False),
        'missing_soft_output': gr.update(visible=False),
        'recommendation_output': gr.update(visible=False),
        'tabs': gr.update(selected=1)
    }

    progress(0, desc="Starting analysis...")

    # Validate CV file
    if not cv_file or not cv_file.name.endswith('.pdf'):
        default_output['report_header'] = gr.update(visible=True, value="## Error\n---\nPlease upload a valid PDF file.")
        yield list(default_output.values())
        return

    # Validate job description
    try:
        job_description = validate_input(job_description, "Job Description")
    except ValueError as e:
        default_output['report_header'] = gr.update(visible=True, value=f"## Error\n---\n{e}")
        yield list(default_output.values())
        return

    # Show loading state
    loading_updates = {
        'report_header': gr.update(visible=True, value="### Aurora is thinking... Analyzing your profile."),
        'matched_tech_header': gr.update(visible=False),
        'matched_tech_output': gr.update(visible=False),
        'matched_soft_header': gr.update(visible=False),
        'matched_soft_output': gr.update(visible=False),
        'missing_tech_header': gr.update(visible=False),
        'missing_tech_output': gr.update(visible=False),
        'missing_soft_header': gr.update(visible=False),
        'missing_soft_output': gr.update(visible=False),
        'recommendation_output': gr.update(visible=False),
        'tabs': gr.update(selected=1)
    }
    yield list(loading_updates.values())

    try:
        progress(0.2, desc="Extracting text from CV...")
        pdf_reader = PyPDF2.PdfReader(cv_file.name)
        cv_text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

        progress(0.4, desc="Identifying required skills...")
        job_prompt = f"""
        Analyze the following job description and identify the required technical and soft skills.
        Return a JSON object with two lists: `required_technical_skills` and `required_soft_skills`.
        Job Description: {job_description}
        """
        job_response = call_gemini_api(job_prompt)
        job_result = safe_parse_json(job_response.text)
        required_tech = job_result.get('required_technical_skills', [])
        required_soft = job_result.get('required_soft_skills', [])

        progress(0.6, desc="Matching skills in CV...")
        cv_prompt = f"""
        Analyze the following CV text and identify which of these technical skills ({', '.join(required_tech)})
        and soft skills ({', '.join(required_soft)}) are present. Return a JSON object with two lists:
        `matched_technical_skills` and `matched_soft_skills`.
        CV Text: {cv_text}
        """
        cv_response = call_gemini_api(cv_prompt)
        cv_result = safe_parse_json(cv_response.text)
        matched_tech_skills = cv_result.get('matched_technical_skills', [])
        matched_soft_skills = cv_result.get('matched_soft_skills', [])

        progress(0.8, desc="Calculating match score...")
        missing_tech_skills = [skill for skill in required_tech if skill not in matched_tech_skills]
        missing_soft_skills = [skill for skill in required_soft if skill not in matched_soft_skills]

        total_required = len(required_tech) + len(required_soft)
        total_matched = len(matched_tech_skills) + len(matched_soft_skills)
        match_score = (total_matched / total_required) * 100 if total_required > 0 else 0

        def create_skill_tags(skill_list):
            return "".join([f"<span class='skill-tag'>{skill}</span>" for skill in sorted(skill_list)])

        matched_tech_html = create_skill_tags(matched_tech_skills) if matched_tech_skills else "<p>No technical skills found</p>"
        matched_soft_html = create_skill_tags(matched_soft_skills) if matched_soft_skills else "<p>No soft skills found</p>"

        recommendation_text = ""
        if missing_tech_skills or missing_soft_skills:
            recommendation_text += "### Your Recommended Learning Path\n"
            if missing_tech_skills:
                recommendation_text += f"- Focus on these technical skills: **{', '.join(sorted(missing_tech_skills))}**.\n"
            if missing_soft_skills:
                recommendation_text += f"- Focus on these soft skills: **{', '.join(sorted(missing_soft_skills))}**.\n"
            recommendation_text += "- We suggest exploring platforms like **Coursera, freeCodeCamp, and Google Skillshop** to master these areas."
        else:
            recommendation_text = "\n\n**Congratulations! You are a strong candidate for this role!**"

        progress(1.0, desc="Analysis complete!")
        final_report = {
            'report_header': gr.update(visible=True, value=f"## Aurora Analysis Report\n---\n**Overall Match Score:** {match_score:.2f}%"),
            'matched_tech_header': gr.update(visible=True),
            'matched_tech_output': gr.update(visible=True, value=matched_tech_html),
            'matched_soft_header': gr.update(visible=True),
            'matched_soft_output': gr.update(visible=True, value=matched_soft_html),
            'missing_tech_header': gr.update(visible=True if missing_tech_skills else False, value="### Missing Technical Skills"),
            'missing_tech_output': gr.update(visible=True if missing_tech_skills else False, value=create_skill_tags(missing_tech_skills) if missing_tech_skills else "<p>No missing technical skills</p>"),
            'missing_soft_header': gr.update(visible=True if missing_soft_skills else False, value="### Missing Soft Skills"),
            'missing_soft_output': gr.update(visible=True if missing_soft_skills else False, value=create_skill_tags(missing_soft_skills) if missing_soft_skills else "<p>No missing soft skills</p>"),
            'recommendation_output': gr.update(visible=True, value=recommendation_text),
            'tabs': gr.update(selected=1)
        }
        yield list(final_report.values())

    except Exception as e:
        logging.error(f"Error in analyze_cv: {e}")
        default_output['report_header'] = gr.update(visible=True, value=f"## An Error Occurred\n---\n {e}")
        yield list(default_output.values())

def generate_learning_path(missing_skills, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Starting learning path generation...")
    try:
        missing_skills = validate_input(missing_skills, "Missing Skills")
        yield gr.update(visible=True, value="### Generating Learning Path...")
        time.sleep(1)  # Simulate processing
        skills_list = [s.strip() for s in missing_skills.split(',')]
        progress(0.5, desc="Querying recommendations...")
        prompt = f"""
        For the following missing skills: {', '.join(skills_list)}, suggest specific online courses or resources
        from platforms like Coursera, freeCodeCamp, or Google Skillshop. Return a JSON object with a list of
        recommendations, each containing `skill`, `platform`, and `course_name`.
        """
        response = call_gemini_api(prompt)
        result = safe_parse_json(response.text)
        recommendations = result.get('recommendations', [])

        output = "### Personalized Learning Path\n"
        for rec in recommendations:
            output += f"- **{rec['skill']}**: {rec['course_name']} on {rec['platform']}\n"

        progress(1.0, desc="Generation complete!")
        yield gr.update(visible=True, value=output if recommendations else "No specific recommendations found.")
    except Exception as e:
        logging.error(f"Error generating learning path: {e}")
        yield gr.update(visible=True, value=f"Error generating learning path: {e}")

def generate_interview_questions(job_description, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Starting interview questions generation...")
    try:
        job_description = validate_input(job_description, "Job Description")
        yield gr.update(visible=True, value="### Generating Interview Questions...")
        time.sleep(1)  # Simulate processing
        progress(0.5, desc="Querying questions...")
        prompt = f"""
        Based on the following job description, generate 3 relevant interview questions for the role.
        Return a JSON object with a list of questions.
        Job Description: {job_description}
        """
        response = call_gemini_api(prompt)
        result = safe_parse_json(response.text)
        questions = result.get('questions', [])

        output = "### Interview Questions\n"
        for i, question in enumerate(questions, 1):
            output += f"{i}. {question}\n"

        progress(1.0, desc="Generation complete!")
        yield gr.update(visible=True, value=output if questions else "No questions generated.")
    except Exception as e:
        logging.error(f"Error generating interview questions: {e}")
        yield gr.update(visible=True, value=f"Error generating interview questions: {e}")

# --- UI SECTION ---
aurora_theme = gr.Theme(
    primary_hue="orange", secondary_hue="amber", neutral_hue="stone",
    font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("Sansation"), "ui-monospace", "monospace")
).set(
    body_background_fill_dark="#000000",
    button_primary_background_fill="#ff580f",
    button_primary_background_fill_hover="#ff7e26",
    button_secondary_background_fill="#ffa472",
    button_secondary_background_fill_hover="#ff9c59"
)

custom_css = """
/* Ensure transparent background for Gradio container */
.aurora-gradio-container { background: transparent !important; }

/* Starry background with irregular shining stars */
.background-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    background: #000000;
}
.star-layer {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
}
.star-layer-1 {
    background:
        radial-gradient(circle, #ffffff 1px, transparent 1px) 10% 15% / 50px 50px,
        radial-gradient(circle, #ffffff 2px, transparent 2px) 20% 30% / 60px 60px,
        radial-gradient(circle, #ffffff 1.5px, transparent 1.5px) 35% 45% / 40px 40px;
    opacity: 0.4;
    animation: twinkle 4s infinite;
}
.star-layer-2 {
    background:
        radial-gradient(circle, #ffffff 1px, transparent 1px) 5% 20% / 55px 55px,
        radial-gradient(circle, #ffffff 2px, transparent 2px) 30% 40% / 70px 70px;
    opacity: 0.35;
    animation: twinkle 5s infinite 1.5s;
}
@keyframes twinkle {
    0% { opacity: 0.2; }
    50% { opacity: 0.8; }
    100% { opacity: 0.2; }
}

/* Sidebar tab buttons */
div.tabitem button, .tabs button {
    background-color: #000000 !important;
    color: #ff7e26 !important;
    padding: 5px 12px;
    border-radius: 15px;
    margin: 4px;
    display: inline-block;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.3s ease-in-out;
}
div.tabitem button:hover, .tabs button:hover {
    background-color: #000000 !important;
    transform: scale(1.05);
    box-shadow: 0 0 12px #ff9c59;
}
div.tabitem button[aria-selected="true"], .tabs button[aria-selected="true"] {
    background-color: #000000 !important;
    color: #ff7e26 !important;
}
.dark div.tabitem button, .dark .tabs button {
    background-color: #000000 !important;
    color: #ff7e26 !important;
}
.dark div.tabitem button:hover, .dark .tabs button:hover {
    background-color: #000000 !important;
    box-shadow: 0 0 12px #ff9c59;
}
.dark div.tabitem button[aria-selected="true"], .dark .tabs button[aria-selected="true"] {
    background-color: #000000 !important;
    color: #ff7e26 !important;
}

/* Pop-up effect for primary buttons */
button.primary {
    transition: all 0.3s ease-in-out;
}
button.primary:hover {
    box-shadow: 0 0 15px #ff7e26, 0 0 25px #ff7e26;
    transform: translateY(-2px);
}

/* Styling for interactive elements */
.gradio-container .primary, .skill-tag, div.tabitem button, .tabs button {
    transition: all 0.3s ease-in-out;
}
.skill-tag:hover {
    transform: scale(1.05);
    box-shadow: 0 0 12px #ffa472;
}
.report-section-header {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 5px;
}
.skill-tag {
    background-color: #ffc99d;
    color: #2a1c0f;
    padding: 5px 12px;
    border-radius: 15px;
    margin: 4px;
    display: inline-block;
    font-size: 0.9rem;
    font-weight: 500;
}
.dark .skill-tag {
    background-color: #000000;
    color: #ff7e26;
}
"""

with gr.Blocks(theme=aurora_theme, css=custom_css, elem_classes="aurora-gradio-container") as interface:
    gr.HTML("""
        <div class="background-container">
            <div class="star-layer star-layer-1"></div>
            <div class="star-layer star-layer-2"></div>
        </div>
    """)

    with gr.Row():
        gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: center; width: 100%;">
                <h1 style="font-family: 'Sansation', monospace; font-weight: 400; font-size: 2.5rem; margin: 20px; letter-spacing: 0.1em;">AURORA</h1>
            </div>
        """)

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs(elem_classes="tabitem") as tabs:
                with gr.TabItem("Homepage", id=0):
                    gr.Markdown("## Welcome to Aurora")
                    gr.Markdown("**Benchmark** your skills, **Elevate** your profile, and **Ace** the interview with Aurora.")

                with gr.TabItem("Benchmark", id=1):
                    gr.Markdown("## Benchmark Your Skills\nUpload your CV and provide a job title or description to analyze your technical and soft skills.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            cv_input = gr.File(label="1. Upload Your CV (PDF only)")
                            job_description_input = gr.Textbox(label="2. Enter Your Job Description", placeholder="e.g., Data Analyst, Software Engineer, or paste a job description")
                            analyze_button = gr.Button("Analyze Now", variant="primary")
                        with gr.Column(scale=2):
                            report_header = gr.Markdown(visible=False, elem_id="report_header")
                            matched_tech_header = gr.Markdown("### Matched Technical Skills", visible=False, elem_classes="report-section-header")
                            matched_tech_output = gr.HTML(visible=False, elem_id="matched_tech_output")
                            matched_soft_header = gr.Markdown("### Matched Soft Skills", visible=False, elem_classes="report-section-header")
                            matched_soft_output = gr.HTML(visible=False, elem_id="matched_soft_output")
                            missing_tech_header = gr.Markdown("### Missing Technical Skills", visible=False, elem_classes="report-section-header")
                            missing_tech_output = gr.HTML(visible=False, elem_id="missing_tech_output")
                            missing_soft_header = gr.Markdown("### Missing Soft Skills", visible=False, elem_classes="report-section-header")
                            missing_soft_output = gr.HTML(visible=False, elem_id="missing_soft_output")
                            recommendation_output = gr.Markdown(visible=False, elem_id="recommendation_output")

                with gr.TabItem("Elevate", id=2):
                    gr.Markdown("## Elevate Your Profile\nGet personalized learning paths to build the skills you need to succeed.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            missing_skills_input = gr.Textbox(label="Enter Missing Skills (comma-separated)", placeholder="e.g., SQL, Python, Communication")
                            learning_path_button = gr.Button("Generate Learning Path", variant="primary")
                        with gr.Column(scale=2):
                            learning_path_output = gr.Markdown(visible=False, elem_id="learning_path_output")

                with gr.TabItem("Ace", id=3):
                    gr.Markdown("## Ace the Interview\nPractice with AI-generated interview questions tailored to your job role.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            interview_job_input = gr.Textbox(label="Enter Job Title or Description", placeholder="e.g., Data Analyst, Software Engineer")
                            interview_button = gr.Button("Generate Interview Questions", variant="primary")
                        with gr.Column(scale=2):
                            interview_output = gr.Markdown(visible=False, elem_id="interview_output")

    analyze_button.click(
        fn=analyze_cv,
        inputs=[cv_input, job_description_input],
        outputs=[
            report_header, matched_tech_header, matched_tech_output,
            matched_soft_header, matched_soft_output, missing_tech_header,
            missing_tech_output, missing_soft_header, missing_soft_output,
            recommendation_output, tabs
        ]
    )
    learning_path_button.click(
        fn=generate_learning_path,
        inputs=[missing_skills_input],
        outputs=[learning_path_output]
    )
    interview_button.click(
        fn=generate_interview_questions,
        inputs=[interview_job_input],
        outputs=[interview_output]
    )

interface.launch()