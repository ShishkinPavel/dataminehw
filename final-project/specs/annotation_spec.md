# Спецификация разметки: sentiment_classification

## Задача
Sentiment classification текстовых отзывов. Задача — определить общий эмоциональный тон текста (positive/negative).

## Классы

### positive
**Определение:** Текст выражает положительное отношение, одобрение, удовлетворение.
**Примеры:**
1. "Now I'm not saying I know how to play this game. All I'm saying is that all your settlements are belong to me"
2. "Are you tired of being called a Freak?!  Do People tell you thing that you don't want to hear?! Well, do they?! If you answered yes, then Normailty is for you. Play this game and you'll see, meet &amp..."
3. "Still to this day, I have no idea how it ended up in my library or how I managed to get this many hours on it."

### negative
**Определение:** Текст выражает отрицательное отношение, критику, неудовлетворённость.
**Примеры:**
1. "Little game, such shame, much disappoint, not amaze."
2. "10/10 GOTY 2014 Pros: -Shoot dead people. -Blow up dead people. -Dead people.  Cons: -Its eSports. -Die from dead people. -Dead people.  Needs workshop support."
3. "This game has a great concept but not much is done with it,It's far too short and not worth the mone time or effort to play it as many questions were left unanswered.The puzzles are either to cryptic ..."

## Граничные случаи
1. **Смешанные отзывы:** "The acting was great but the plot was terrible" — размечать по общему тону
2. **Ирония/сарказм:** "Oh great, another masterpiece..." — negative (несмотря на "great")
3. **Нейтральные:** "The movie was 2 hours long" — если нет явного тона, смотреть контекст
4. **Короткие тексты:** Одно слово/фраза — опираться на коннотацию

## Инструкция для разметчика
1. Прочитайте весь текст целиком
2. Определите общий тон (positive/negative)
3. При неуверенности — отметьте как "uncertain" в комментариях
4. Время на один пример: ~10-15 секунд