import os, sys, argparse
from io import BytesIO

from flask import Flask, render_template, request, redirect, url_for, session, send_file, abort, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import docx2txt, PyPDF2
from fpdf import FPDF

# ML + NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords, wordnet # Keep wordnet import for NLTK's internal use
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# NLTK Data Downloads (Ensures all necessary resources are available)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab') # CRITICAL FIX: Ensure punkt_tab is downloaded

# ── NLP Preprocessing Setup
stop_words = set(stopwords.words('english'))
# IMPORTANT: WordNetLemmatizer will be initialized inside preprocess_text for robustness
# Remove global 'lemmatizer' = WordNetLemmatizer() from here!

# ── Flask & DB Setup
app = Flask(__name__)
app.secret_key = "your_secret_key" # Essential for secure sessions
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///ats_users_new.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False # Disable modification tracking for memory efficiency
db = SQLAlchemy(app)

# ── ML Classifier for Fit Probability (used for single resume analysis)
sample_data = [
    ("python pandas numpy machine learning", 1),
    ("excel powerpoint word communication", 0),
    ("data analysis sql matplotlib seaborn", 1),
    ("data analysis sql matplotlib seaborn", 1),
    ("team work hard work adaptability", 0),
    ("deep learning pytorch keras tensorflow", 1),
    ("leadership presentation decision making", 0),
]
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform([text for text, label in sample_data])
y_train = [label for text, label in sample_data]
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

def predict_fit_probability(resume_text, jd_text):
    """
    Predicts the probability (as a percentage) that a single resume fits a job description
    based on a pre-trained ML model.
    """
    combined = f"{resume_text.lower()} {jd_text.lower()}"
    vector = vectorizer.transform([combined])
    prob = classifier.predict_proba(vector)[0][1] # Probability of the positive class (fit)
    return round(prob * 100, 1)

def rank_resumes(resume_texts_raw, jd_text_raw):
    """
    Ranks multiple resumes against a single job description based on TF-IDF cosine similarity.
    Returns a sorted list of (resume_index, similarity_score).
    """
    # Preprocess all texts: JD first, then resumes
    processed_jd = preprocess_text(jd_text_raw)
    processed_resumes = [preprocess_text(res) for res in resume_texts_raw]

    # Combine all processed texts for TF-IDF vectorization
    all_texts = [processed_jd] + processed_resumes
    
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(all_texts)
    
    jd_vec = matrix[0] # The first row is the JD's TF-IDF vector
    resume_vecs = matrix[1:] # Remaining rows are resume vectors
    
    scores = []
    for i, resume_vec in enumerate(resume_vecs):
        # Calculate cosine similarity between each resume and the JD
        similarity = (resume_vec @ jd_vec.T).toarray()[0][0]
        scores.append((i, similarity)) # Store original index and score
        
    # Sort scores in descending order
    return sorted(scores, key=lambda x: x[1], reverse=True)

# ── DB Models
class User(db.Model):
    """
    SQLAlchemy model for storing user information (username, email, password hash, etc.).
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    mobile = db.Column(db.String(20))
    password_hash = db.Column(db.String(128), nullable=False)

# ── Helper Functions
def preprocess_text(text):
    """
    Cleans and preprocesses text for NLP tasks: tokenization, lowercasing,
    stop word removal, and lemmatization.
    """
    # CRITICAL FIX: Initialize WordNetLemmatizer inside the function
    # This ensures a fresh instance is created on each call, resolving
    # potential lazy-loading or state issues with NLTK's WordNet.
    lemmatizer_local = WordNetLemmatizer()

    tokens = word_tokenize(text.lower())
    cleaned = [lemmatizer_local.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(cleaned)

def extract_text(file_storage_object):
    """
    Extracts text content from various file types (PDF, DOC/DOCX, TXT).
    Handles Flask's FileStorage objects.
    """
    ext = file_storage_object.filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        reader = PyPDF2.PdfReader(file_storage_object)
        return "".join(page.extract_text() or "" for page in reader.pages)
    elif ext in ["doc", "docx"]:
        # docx2txt expects a file-like object or path; FileStorage is file-like
        return docx2txt.process(file_storage_object)
    # Default to reading as plain text for any other file type
    return file_storage_object.read().decode("utf-8", errors="ignore")

def calculate_score(resume_text, jd_text):
    """
    Calculates a similarity score (0-100) between a single resume and a JD
    using TF-IDF cosine similarity.
    """
    corpus = [resume_text, jd_text]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(corpus)
    # Cosine similarity between the first document (resume) and second (JD)
    similarity = (matrix[0] @ matrix[1].T).toarray()[0][0]
    return int(similarity * 100)

def pdf_report(score, matched, missing):
    """
    Generates a PDF report summarizing ATS analysis results for a single resume.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "ATS Score Report", ln=True, align="C")
    pdf.ln(6)
    pdf.cell(0, 10, f"Match Score: {score}%", ln=True)
    pdf.ln(4)
    # Ensure matched and missing are iterable (lists) even if empty
    matched_str = ", ".join(matched) if matched else "None"
    missing_str = ", ".join(missing) if missing else "None"
    pdf.multi_cell(0, 8, f"Matched Keywords:\n{matched_str}")
    pdf.ln(4)
    pdf.multi_cell(0, 8, f"Missing Keywords:\n{missing_str}")
    
    # Output PDF to a BytesIO object to send as a file
    buf = BytesIO()
    # It's better to use pdf.output(dest='S') to get bytes and then write
    pdf_content = pdf.output(dest='S').encode('latin1')
    buf.write(pdf_content)
    buf.seek(0) # Rewind to the beginning of the buffer
    return buf

# ── Routes
@app.route("/")
def index():
    """Renders the main homepage."""
    return render_template("index.html", username=session.get("username"))

@app.route("/dashboard")
def dashboard():
    """Renders the user dashboard, requiring login."""
    if "username" not in session:
        flash("Please log in to access dashboard.", "warning")
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])

@app.route("/select_role", methods=["POST"])
def select_role():
    """
    Handles role selection from dashboard and redirects to appropriate upload page.
    Candidate -> single resume upload (`upload.html`)
    HR -> multi-resume upload (`rank_upload.html`)
    """
    if "username" not in session: # Ensure user is logged in before allowing role selection
        flash("Please log in to proceed.", "warning")
        return redirect(url_for("login"))

    role = request.form.get("role")
    if role == "candidate":
        session["role"] = "candidate"
        flash("You have selected to analyze a single resume. Please upload your documents.", "info")
        return redirect(url_for("upload_page"))
    elif role == "hr":
        session["role"] = "hr"
        flash("You have selected to rank multiple resumes. Please upload your job description and resumes.", "info")
        return redirect(url_for("rank_upload_page")) # Redirect to new HR specific upload page
    else:
        flash("Invalid role selection.", "danger")
        return redirect(url_for("dashboard"))

@app.route("/upload_page")
def upload_page():
    """Renders the single resume upload form page (for Candidates)."""
    if "username" not in session:
        flash("Please log in to upload files.", "warning")
        return redirect(url_for("login"))
    # Ensure the correct template is rendered for candidates
    return render_template("upload.html")

@app.route("/rank_upload_page")
def rank_upload_page():
    """Renders the multi-resume upload form page (for HR)."""
    if "username" not in session:
        flash("Please log in to upload files.", "warning")
        return redirect(url_for("login"))
    # Ensure the correct template is rendered for HR
    return render_template("rank_upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles single resume file upload from 'upload.html', performs ATS analysis,
    and renders 'result.html'. This route is for Candidates.
    """
    if "username" not in session:
        flash("Please log in to upload files.", "warning")
        return redirect(url_for("login"))

    resume_file = request.files.get("resume")
    jd_file = request.files.get("job_description") # Name changed to job_description in upload.html

    if not resume_file or not jd_file:
        flash("Please upload both a Resume and a Job Description file.", "danger")
        return redirect(url_for("upload_page")) # Redirect back to single upload page

    resume_text_processed = preprocess_text(extract_text(resume_file))
    jd_text_processed = preprocess_text(extract_text(jd_file))
    
    score = calculate_score(resume_text_processed, jd_text_processed)
    fit_prob = predict_fit_probability(resume_text_processed, jd_text_processed)
    
    # Determine matched and missing keywords
    resume_tokens = set(resume_text_processed.split())
    jd_tokens = set(jd_text_processed.split())
    matched = list(resume_tokens.intersection(jd_tokens))
    missing = list(jd_tokens.difference(resume_tokens))
    
    # Store results in session for PDF download
    session.update({
        "score": score,
        "matched": matched,
        "missing": missing,
        "fit_prob": fit_prob
    })
    flash("Resume analyzed successfully!", "success")
    return render_template("result.html",
                           score=score,
                           fit_prob=fit_prob,
                           matched_keywords=matched,
                           missing_keywords=missing)

@app.route("/rank_process", methods=["POST"])
def rank_process():
    """
    Handles multiple resume uploads from 'rank_upload.html', performs ranking,
    and renders 'ranking_result.html'. This route is for HRs.
    """
    if "username" not in session:
        flash("Please log in to access this feature.", "warning")
        return redirect(url_for("login"))

    jd_file = request.files.get("jd") # Name 'jd' from rank_upload.html
    resume_files = request.files.getlist("resumes") # Name 'resumes' from rank_upload.html

    if not jd_file or not resume_files:
        flash("Please upload a Job Description and at least one Resume file for ranking.", "danger")
        return redirect(url_for("rank_upload_page"))

    raw_resume_texts = [extract_text(r) for r in resume_files]
    raw_jd_text = extract_text(jd_file)

    ranked_scores_with_indices = rank_resumes(raw_resume_texts, raw_jd_text)
    
    # Prepare results for the ranking_result.html template
    results_for_template = []
    for index, score in ranked_scores_with_indices:
        results_for_template.append({
            "filename": resume_files[index].filename, # Use original filename
            "score": round(score * 100, 2) # Convert similarity score to percentage
        })
    
    flash(f"Analyzed {len(resume_files)} resumes and ranked them against the JD.", "success")
    return render_template("ranking_result.html", results=results_for_template)


@app.route("/download_pdf")
def download_pdf():
    """Allows downloading the detailed PDF report for a single resume analysis."""
    # Ensure necessary session data exists for PDF generation
    if "score" not in session or "matched" not in session or "missing" not in session:
        flash("No report data found. Please perform a single resume analysis first.", "warning")
        return redirect(url_for("upload_page")) # Redirect if no data to report

    pdf_buffer = pdf_report(session.get("score", 0), session.get("matched", []), session.get("missing", []))
    return send_file(pdf_buffer, download_name="ATS_Report.pdf", as_attachment=True, mimetype='application/pdf')

@app.route("/admin")
def admin_panel():
    """Renders the admin panel, restricted to the 'admin2' user."""
    if session.get("username") != "admin2":
        abort(403) # Forbidden
    users = User.query.all() # Fetch all registered users
    return render_template("admin.html", users=users, total=len(users))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Handles user registration."""
    if request.method == "POST":
        uname = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        mobile = request.form["mobile"].strip()
        pwd = generate_password_hash(request.form["password"])
        
        # Consolidated validation and flash messages
        if uname.lower() == "admin":
            flash("Cannot use reserved username 'admin'.", "danger")
            return redirect(url_for("signup"))
        if User.query.filter_by(username=uname).first():
            flash("Username already exists.", "danger")
            return redirect(url_for("signup"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for("signup"))
            
        db.session.add(User(username=uname, email=email, mobile=mobile, password_hash=pwd))
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handles user login."""
    if request.method == "POST":
        uname = request.form["username"]
        pwd = request.form["password"]
        user = User.query.filter_by(username=uname).first()
        if user and check_password_hash(user.password_hash, pwd):
            session["username"] = uname
            session["is_admin"] = uname == "admin2"
            flash(f"Welcome back, {uname}!", "success")
            return redirect(url_for("index"))
        flash("Invalid credentials. Please try again.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    """Logs out the current user by clearing the session."""
    session.clear()
    flash("You have been successfully logged out.", "info")
    return redirect(url_for("index"))

# ── DB Initialization & Admin Seeder
def init_db():
    """Initializes the database by creating all tables."""
    with app.app_context(): # Ensure this runs within the Flask application context
        db.create_all()
        print("✅ Database initialized")

def create_default_admin():
    """Creates a default admin user if one doesn't exist."""
    with app.app_context(): # Ensure this runs within the Flask application context
        if not User.query.filter_by(username='admin2').first():
            admin = User(username='admin2', email='admin2@example.com', mobile='9876543210', password_hash=generate_password_hash("admin@987"))
            db.session.add(admin)
            db.session.commit()
            print("✅ Default admin user created: admin2 / admin@987")

# ── Main Application Runner
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ATS Flask Application or initialize the database.")
    parser.add_argument("--init-db", action="store_true", help="Initialize the database and create a default admin user, then exit.")
    args = parser.parse_args()

    if args.init_db:
        init_db()
        create_default_admin()
        sys.exit(0) # Exit after database initialization

    # Initialize DB and create admin if DB file does not exist when app starts normally
    db_path = "ats_users_new.db"
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' not found. Initializing database and creating default admin...")
        init_db()
        create_default_admin()
    else:
        print(f"Database file '{db_path}' found. Skipping initialization.")


    print("Starting Flask application...")
    # Run the Flask application in debug mode for development.
    # Turn debug=False for production deployments.
    app.run(debug=True)
