# Movie Explorer – поиск фильмов, близких по описанию

В `storage/new_better_descriptions.txt` сохранены названия и развернутые описания ~1.5k фильмов с IMDb. В `storage/imdb_model` сохранена модель `FastText`. `main.py` – запуск локального сервера `Flask`.

### Установка:

```
pip install -r requirements.txt
python main.py
```

### Демонстрация работы:

- Открыть [localhost](http://127.0.0.1:5000)
- Можно выбрать фильм из выпадающего списка или самостоятельно ввести название *на английском* 