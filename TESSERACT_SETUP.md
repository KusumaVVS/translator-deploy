# Tesseract Path Configuration Guide

This guide explains how to configure a custom Tesseract OCR path for the Translator project.

## Important: Path Format

**For Windows:** Set `TESSERACT_PATH` to the **directory** containing `tesseract.exe`, NOT the executable itself.

**For Linux/macOS:** Set `TESSERACT_PATH` to the **full path** to the `tesseract` executable.

### Examples:
- ✅ **Correct (Windows):** `C:\Program Files\Tesseract-OCR`
- ❌ **Wrong (Windows):** `C:\Program Files\Tesseract-OCR\tesseract.exe`
- ✅ **Correct (Linux/macOS):** `/usr/bin/tesseract`

## Method 1: Windows Command Prompt (Temporary)

Set the environment variable for the current session:

```cmd
set TESSERACT_PATH=C:\Program Files\Tesseract-OCR
python app.py
```

**Note:** This only works for the current command prompt session. You'll need to set it again if you open a new terminal.

## Method 2: Windows PowerShell (Temporary)

```powershell
$env:TESSERACT_PATH = "C:\Program Files\Tesseract-OCR"
python app.py
```

## Method 3: Windows System Environment Variables (Permanent)

### Option A: Using System Properties

1. Press `Win + R` and type `sysdm.cpl`
2. Click the "Advanced" tab
3. Click "Environment Variables"
4. Under "User variables" or "System variables", click "New"
5. Variable name: `TESSERACT_PATH`
6. Variable value: `C:\Program Files\Tesseract-OCR` (directory path, not including tesseract.exe)
7. Click OK on all dialogs
8. Restart your terminal/IDE

### Option B: Using Command Prompt (Permanent)

```cmd
setx TESSERACT_PATH "C:\Program Files\Tesseract-OCR"
```

**Note:** This sets the variable permanently for your user account. You'll need to restart your terminal for changes to take effect.

## Method 4: Linux/macOS (Temporary)

```bash
export TESSERACT_PATH=/usr/bin/tesseract
python app.py
```

## Method 5: Linux/macOS (Permanent)

Add to your shell configuration file:

### For Bash (~/.bashrc or ~/.bash_profile):
```bash
echo 'export TESSERACT_PATH=/usr/bin/tesseract' >> ~/.bashrc
source ~/.bashrc
```

### For Zsh (~/.zshrc):
```bash
echo 'export TESSERACT_PATH=/usr/bin/tesseract' >> ~/.zshrc
source ~/.zshrc
```

## Method 6: Python Script (Runtime)

You can also set the path directly in your Python script before importing:

```python
import os
os.environ['TESSERACT_PATH'] = r'C:\Program Files\Tesseract-OCR'

# Then import your modules
from modules.image_ocr import ocr_from_image_bytes
```

## Method 7: .env File (Recommended for Development)

Create a `.env` file in your project root:

```env
TESSERACT_PATH=C:\Program Files\Tesseract-OCR
```

Then load it in your Python script:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

# The TESSERACT_PATH will now be available
```

**Note:** You'll need to install python-dotenv first:
```bash
pip install python-dotenv
```

## Common Tesseract Installation Paths

### Windows (Directory paths):
- `C:\Program Files\Tesseract-OCR`
- `C:\Program Files (x86)\Tesseract-OCR`
- `C:\Tesseract-OCR`

### Linux (Full executable paths):
- `/usr/bin/tesseract`
- `/usr/local/bin/tesseract`

### macOS (Homebrew) (Full executable paths):
- `/usr/local/bin/tesseract`
- `/opt/homebrew/bin/tesseract` (Apple Silicon)

## Verification

To verify your Tesseract installation and path:

### Windows:
```cmd
"C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

### Linux/macOS:
```bash
tesseract --version
```

## Troubleshooting

### Issue: "Tesseract not found" or "FileNotFoundError"

**Solution:** 
1. Verify Tesseract is installed
2. Check the path is correct
3. **For Windows:** Ensure the path points to the **directory** containing `tesseract.exe`, not the executable itself
4. **For Linux/macOS:** Ensure the path points to the **executable** file

### Issue: "Permission denied"

**Solution:**
- Run your terminal/IDE as Administrator (Windows)
- Check file permissions on the Tesseract directory

### Issue: Path not recognized

**Solution:**
- Use raw strings in Python: `r'C:\Path\To\Tesseract'`
- Use forward slashes: `'C:/Path/To/Tesseract'`
- Escape backslashes: `'C:\\Path\\To\\Tesseract'`

### Issue: "Custom Tesseract path not found, using default"

**Solution:**
- The custom path you set doesn't exist
- Check for typos in the path
- Verify the directory actually exists on your system

## Default Behavior

If `TESSERACT_PATH` is not set or the custom path doesn't exist, the application will use the default Windows path:
```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

The application will now:
1. Check if `TESSERACT_PATH` is set
2. Verify the path exists
3. If the custom path doesn't exist, fall back to the default path
4. Display a warning if using the fallback

## Example Usage

### Windows Command Prompt:
```cmd
set TESSERACT_PATH=C:\Program Files\Tesseract-OCR
python app.py
```

### Windows PowerShell:
```powershell
$env:TESSERACT_PATH = "C:\Program Files\Tesseract-OCR"
python app.py
```

### Linux/macOS:
```bash
export TESSERACT_PATH=/usr/bin/tesseract
python app.py
```

## Testing Your Configuration

Run the test script to verify your Tesseract configuration:

```bash
python test_image_preprocessing.py
```

If Tesseract is configured correctly, you should see:
- A message indicating which Tesseract path is being used
- OCR results from the test images

## Unsetting Custom Path

If you want to remove a custom Tesseract path and use the default:

### Windows Command Prompt:
```cmd
set TESSERACT_PATH=
python app.py
```

### Windows PowerShell:
```powershell
$env:TESSERACT_PATH = ""
python app.py
```

### Linux/macOS:
```bash
unset TESSERACT_PATH
python app.py
```
