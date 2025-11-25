# AJ Supplements Software

Django-based web application for supplement management.

## Prerequisites for Windows

1. **Python 3.13 or 3.14** - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   
2. **PostgreSQL** (Optional - if using the remote database, you may not need local PostgreSQL)
   - Download from [postgresql.org](https://www.postgresql.org/download/windows/)

## Setup Instructions for Windows

### Step 1: Extract/Clone the Repository

If you downloaded as ZIP:
- Extract the ZIP file to a location like `C:\Users\YourName\Desktop\ajsupplements_software`

If using Git:
```bash
git clone https://github.com/shaiksajidhussain/ajsupplements_software.git
cd ajsupplements_software
```

### Step 2: Navigate to the Project Folder

Open Command Prompt or PowerShell and navigate to the project:
```bash
cd "C:\Users\YourName\Desktop\ajsupplements_software\software - Copy - Copy"
```

### Step 3: Activate the Virtual Environment

**For Command Prompt (CMD):**
```bash
venv\Scripts\activate.bat
```

**For PowerShell:**
```bash
venv\Scripts\Activate.ps1
```

If you get an execution policy error in PowerShell, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**For Git Bash:**
```bash
source venv/Scripts/activate
```

You should see `(venv)` at the beginning of your command prompt when activated.

### Step 4: Navigate to the Feed Directory

```bash
cd feed
```

### Step 5: Install Dependencies (if needed)

The virtual environment already contains all dependencies, but if you encounter issues, you can reinstall:

```bash
pip install django==5.2.7
pip install psycopg2-binary
```

### Step 6: Run Database Migrations

```bash
python manage.py migrate
```

### Step 7: Create a Superuser (Optional)

To access the Django admin panel:
```bash
python manage.py createsuperuser
```

Follow the prompts to create an admin user.

### Step 8: Run the Development Server

```bash
python manage.py runserver
```

The server will start at `http://127.0.0.1:8000/`

Open your browser and navigate to:
- **Home Page:** http://127.0.0.1:8000/
- **Admin Panel:** http://127.0.0.1:8000/admin/
- **Login Page:** http://127.0.0.1:8000/login/

## Quick Start Summary (Copy-Paste Commands)

```bash
# 1. Navigate to project folder
cd "C:\path\to\ajsupplements_software\software - Copy - Copy"

# 2. Activate virtual environment
venv\Scripts\activate.bat

# 3. Go to feed directory
cd feed

# 4. Run migrations
python manage.py migrate

# 5. Start server
python manage.py runserver
```

## Troubleshooting

### Issue: "python is not recognized"
- Make sure Python is installed and added to PATH
- Try using `py` instead of `python`: `py manage.py runserver`

### Issue: Virtual environment activation fails
- Make sure you're in the correct directory
- Try using the full path: `C:\path\to\project\venv\Scripts\activate.bat`

### Issue: Database connection error
- The project uses a remote PostgreSQL database
- Make sure you have internet connection
- Database credentials are in `feed/feed/settings.py`

### Issue: Module not found errors
- Make sure the virtual environment is activated (you should see `(venv)` in your prompt)
- Try reinstalling dependencies: `pip install -r requirements.txt` (if requirements.txt exists)

## Project Structure

```
software - Copy - Copy/
├── venv/              # Virtual environment (already set up)
├── feed/              # Django project
│   ├── core/         # Main app
│   ├── feed/         # Project settings
│   ├── manage.py     # Django management script
│   └── db.sqlite3    # SQLite database (if used)
└── .idea/            # IDE settings
```

## Notes

- The virtual environment (`venv`) is already included with all dependencies
- The project is configured to use a remote PostgreSQL database
- All settings are in `feed/feed/settings.py`
