# Flask Translator Deployment Plan - Render.com ✅ APPROVED

## Phase 1: Codebase Preparation ✅ COMPLETE
- ✅ Edit app.py: Production Tesseract paths + cross-platform support
- ✅ Update requirements.txt: Added gunicorn==21.2.0 + pinned versions
- ✅ Create Procfile: web: gunicorn app:app --workers 1 --timeout 120 --preload
- ✅ Create runtime.txt: python-3.11.9
- ✅ Create .gitignore: ML caches, temp files, envs

## Phase 2: GitHub + Render Deploy [PENDING]
1. Create GitHub repo: `gh repo create translator-app --public --push`
2. Test gunicorn locally: `pip install -r requirements.txt && gunicorn app:app`
3. Deploy on Render.com (free web service from GitHub)

## Phase 3: Test & Optimize [PENDING]
- [ ] Live text/image/speech translation
- [ ] Tesseract OCR (set env TESSERACT_PATH on Render)
- [ ] Model pre-caching

## Local Test Commands:
```
pip install -r requirements.txt
gunicorn app:app  # Test production server
pytest tests/ -v  # Run tests
```


