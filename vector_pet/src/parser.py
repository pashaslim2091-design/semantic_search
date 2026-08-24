import re
import json
import os

def parse_ogegov(file_path: str, output_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    articles = text.split('\n\n')
    parsed = []
    for art in articles:
        if not art.strip():
            continue
        first_line = art.split('\n')[0]
        match = re.match(r'^([А-ЯЁа-яё\-]+)', first_line)
        if match:
            word = match.group(1)
        else:
            word = first_line.split(',')[0].strip()
        parsed.append({
            'word': word,
            'full_text': art
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    print(f'Сохранил {len(parsed)} статей в {output_path}')
    return len(parsed)

if __name__ == '__main__':
    parse_ogegov('../data/ogegov.txt', '../data/articles.json')
