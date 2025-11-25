"""
utils.py - miscellaneous helper functions
"""
def save_text(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
